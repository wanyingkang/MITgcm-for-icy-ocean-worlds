#include "SHELFICE_OPTIONS.h"
#ifdef ALLOW_COST
# include "COST_OPTIONS.h"
#endif
#ifdef ALLOW_CTRL
# include "CTRL_OPTIONS.h"
#endif

      SUBROUTINE SHELFICE_INIT_FIXED( myThid )
C     *============================================================*
C     | SUBROUTINE SHELFICE_INIT_FIXED
C     | o Routine to initialize SHELFICE parameters and variables.
C     *============================================================*
C     | Initialize SHELFICE parameters and variables.
C     *============================================================*
      IMPLICIT NONE

C     === Global variables ===
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "SHELFICE.h"
#ifdef ALLOW_COST
# include "cost.h"
# include "SHELFICE_COST.h"
#endif /* ALLOW_COST */

C     === Routine arguments ===
C     myThid ::  Number of this instance of SHELFICE_INIT_FIXED
      INTEGER myThid

#ifdef ALLOW_SHELFICE
C     === Local variables ===
C     i, j, bi, bj :: Loop counters
      INTEGER i, j, bi, bj
#ifdef ALLOW_DIAGNOSTICS
      INTEGER       diagNum
      INTEGER       diagMate
      CHARACTER*8   diagName
      CHARACTER*16  diagCode
      CHARACTER*16  diagUnits
      CHARACTER*(80) diagTitle
#endif /* ALLOW_DIAGNOSTICS */
#ifdef ALLOW_SHIFWFLX_CONTROL
      INTEGER k
# ifdef ALLOW_SHIFWFLX_COST_CONTRIBUTION
      _RL dummy
# endif
#endif
C wykang: add spherical harmonics 
      _RL Y00 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL Y20 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL Y40 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL Y22 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL Y42 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL Y44 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL P1 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL P2 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL P3 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL P4 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL P5 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL P6 (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL lat,lon,clat,slat,c2lon,c4lon
      _RL cobl,p2obl,p4obl,p6obl
      _RL HtideModes(1:5)
      _RL HmixbendModes(1:6)
      _RL Ts,thetaSurface_K
c      move to namelist
c      _RL Htide0

C wykang: g(r)
      _RL rho_bulk, rho_out, H_ice
      _RL p_correction


C----------------------------------------------------------------------
C wykang: add spherical harmonics
      DO bj = myByLo(myThid), myByHi(myThid)
       DO bi = myBxLo(myThid), myBxHi(myThid)
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
            lat=yC(i,j,bi,bj) * deg2rad
            lon=xC(i,j,bi,bj) * deg2rad
            clat=cos(lat)
            slat=sin(lat)
            c2lon=cos(2.*lon)
            c4lon=cos(4.*lon)
            Y00(i,j,bi,bj)=sqrt(1./4./PI)
            Y20(i,j,bi,bj)=sqrt(5./4./PI)*(1.5*slat**2.-0.5)
            Y40(i,j,bi,bj)=sqrt(9./4./PI)*
     &       (35./8.*slat**4.-30./8.*slat**2.+3./8.)
            P1(i,j,bi,bj)=slat
            P2(i,j,bi,bj)=sqrt(4.*PI/5.)*Y20(i,j,bi,bj)
            P3(i,j,bi,bj)=2.5*slat**3.-1.5*slat
            P4(i,j,bi,bj)=(35.*slat**4.-30.*slat**2.+3.)/8.0
            P5(i,j,bi,bj)=(63.*slat**5.-70.*slat**3.+15.*slat)/8.0
            P6(i,j,bi,bj)=(231*slat**6.-315.*slat**4.+105.*slat**2.-5.)
     &                      /16.0
            IF ( Nx .NE. 1 .AND. tide2d ) then
                Y22(i,j,bi,bj)=sqrt(5./4./PI/24.)*
     &              (3.-3.*slat**2.)*(2.*c2lon)
                Y42(i,j,bi,bj)=sqrt(9./4./PI/360.)*
     &              (7.5*(7.*slat**2.-1.)*(1.-slat**2.))*(2.*c2lon)
                Y44(i,j,bi,bj)=sqrt(9./4./PI/40320.)*
     &              (105.*clat**4.)*(2.*c4lon)
            ELSE
                Y22(i,j,bi,bj)=0.
                Y42(i,j,bi,bj)=0.
                Y44(i,j,bi,bj)=0.
            ENDIF
           ENDDO
          ENDDO
         ENDDO
        ENDDO
C----------------------------------------------------------------------

C-- wykang: tidal heating (W/m^2) 
c: for 20.8 km thick ice shell, with Ts=70K, total conduction is around
c  48.9 W/m^2. Part of this Hcond is compensated by heating from
c  core, and the rest should be compensated by tidal heating
c  otherwise, the ice shell would keep melting/freezing, and the ocean
c  salinity will drift away. 
c  moved to namelist
c      Htide0=0.0289
c       : relative mode amplitudes, Y20, Y40, Y22, Y42, Y44 respectively
      cobl=cos(obliquity* deg2rad)
      p2obl= (3.*cobl**2. - 1.)/2.
      p4obl= (35.*cobl**4.-30.*cobl**2. + 3.)/8.
      p6obl= (231.*cobl**6.-315.*cobl**4.+105.*cobl**2.-5.)/16.
      HtideModes=(/0.250, 0.0825, -0.0834, -0.0546, -0.0562/)
      IF (addmixbend) THEN
        HmixbendModes=(/0.124,0.196,-0.0199,-0.0656,0.0132,0.0136/)
      ELSE
        HmixbendModes=(/0,0,0,0,0,0/)
      ENDIF
      if (uniHtide) then
          HtideModes(1)=0.0
          HtideModes(2)=0.0
          HtideModes(3)=0.0
          HtideModes(4)=0.0
          HtideModes(5)=0.0
      endif

      DO bj = myByLo(myThid), myByHi(myThid)
       DO bi = myBxLo(myThid), myBxHi(myThid)
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          Htide(i,j,bi,bj) = Htide0 * sqrt(4.*PI) * ( Y00(i,j,bi,bj) 
     &         + HtideModes(1) * Y20(i,j,bi,bj)
     &         + HtideModes(2) * Y40(i,j,bi,bj)
     &         + HtideModes(3) * Y22(i,j,bi,bj)
     &         + HtideModes(4) * Y42(i,j,bi,bj)
     &         + HtideModes(5) * Y44(i,j,bi,bj) )
          print *, '!!! Htide=',Htide(i,j,bi,bj)
          Hmixbend(i,j,bi,bj) = Htide0 * sqrt(4.*PI)
     &       * ( HmixbendModes(1) * Y00(i,j,bi,bj) 
     &         + HmixbendModes(2) * Y20(i,j,bi,bj)
     &         + HmixbendModes(3) * Y40(i,j,bi,bj)
     &         + HmixbendModes(4) * Y22(i,j,bi,bj)
     &         + HmixbendModes(5) * Y42(i,j,bi,bj)
     &         + HmixbendModes(6) * Y44(i,j,bi,bj) )
          ENDDO
         ENDDO
        ENDDO
       ENDDO

       IF (tiltheat .ne. 0.0) THEN
        DO bj = myByLo(myThid), myByHi(myThid)
         DO bi = myBxLo(myThid), myBxHi(myThid)
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            Htide(i,j,bi,bj)=Htide(i,j,bi,bj)*
     &            (1-tiltheat*yC(i,j,bi,bj)/90.0)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDIF
           
           
        thetaSurface_K=SHELFICEthetaSurface+273.15

        DO bj = myByLo(myThid), myByHi(myThid)
         DO bi = myBxLo(myThid), myBxHi(myThid)
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
             IF (meridionalTs .eq. 1) THEN 
                 lat=yC(i,j,bi,bj) * deg2rad
                 if (abs(yC(i,j,bi,bj))>90.0-obliquity) then
                     Ts=thetaSurface_K*(((PI/2.0-abs(lat))**2.0
     &                  +(obliquity*deg2rad)**2.0)/2.0)**0.125
                 else
                     Ts=thetaSurface_K*(cos(lat))**0.25
                 endif
             ELSEIF (meridionalTs .eq. 3) THEN
                 Ts=thetaSurface_K*(1.0-(5.0/8.0)*p2obl*P2(i,j,bi,bj)
     &              -(9.0/64.0)*p4obl*P4(i,j,bi,bj)
     &              -(65.0/1024.0)*p6obl*P6(i,j,bi,bj))**0.25
             ELSE
                 Ts=thetaSurface_K
             ENDIF
            Hcond(i,j,bi,bj)=abs(SHELFICEkappa)*log(Ts
     &        /(273.15+tRef(1)))/H_ice0
           ENDDO
          ENDDO
         ENDDO
        ENDDO
           
       IF( HP1_ice .ne. 0.0 .OR. HP2_ice .ne. 0.0
     &  .OR. HP3_ice .ne. 0.0 .OR. HP4_ice .ne. 0.0 .OR. 
     &  HP5_ice .ne. 0.0 .OR. HP6_ice .ne.0.0) THEN
        DO bj = myByLo(myThid), myByHi(myThid)
         DO bi = myBxLo(myThid), myBxHi(myThid)
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            Htide(i,j,bi,bj)=Htide(i,j,bi,bj)*(1+HP1_ice*P1(i,j,bi,bj)
     &            +HP2_ice*P2(i,j,bi,bj)+HP3_ice*P3(i,j,bi,bj)
     &            +HP4_ice*P4(i,j,bi,bj)+HP5_ice*P5(i,j,bi,bj)
     &            +HP6_ice*P6(i,j,bi,bj))**ptide_ext

            IF (useHtidePsinHcond) THEN
             Hcond(i,j,bi,bj)=Hcond(i,j,bi,bj)*(1+HP1_ice*P1(i,j,bi,bj)
     &            +HP2_ice*P2(i,j,bi,bj)+HP3_ice*P3(i,j,bi,bj)
     &            +HP4_ice*P4(i,j,bi,bj)+HP5_ice*P5(i,j,bi,bj)
     &            +HP6_ice*P6(i,j,bi,bj))**pcond_ext
            ENDIF
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDIF

       IF( HtP1 .ne. 0.0 .OR. HtP2 .ne. 0.0
     &  .OR. HtP3 .ne. 0.0 .OR. HtP4 .ne. 0.0 .OR. 
     &  HtP5 .ne. 0.0 .OR. HtP6 .ne.0.0) THEN
        DO bj = myByLo(myThid), myByHi(myThid)
         DO bi = myBxLo(myThid), myBxHi(myThid)
          DO j = 1-OLy, sNy+OLy
           DO i = 1-OLx, sNx+OLx
            Htide(i,j,bi,bj)=Htide(i,j,bi,bj) 
     &            +HtP1*P1(i,j,bi,bj)+HtP2*P2(i,j,bi,bj)
     &            +HtP3*P3(i,j,bi,bj)+HtP4*P4(i,j,bi,bj)
     &            +HtP5*P5(i,j,bi,bj)+HtP6*P6(i,j,bi,bj)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
       ENDIF

C-- wykang: end tidal heating

#ifdef ALLOW_MNC
C     Initialize MNC variable information for SHELFICE
      IF ( useMNC .AND. (shelfice_tave_mnc.OR.shelfice_dump_mnc)
     &    ) THEN
        CALL SHELFICE_MNC_INIT( myThid )
      ENDIF
#endif /* ALLOW_MNC */

C-----------------------------------------------------------------------
C--   Initialize SHELFICE variables kTopC
C--   kTopC is the same as kSurfC, except outside ice-shelf area:
C--   kTop = 0 where there is no ice-shelf (where kSurfC=1)
C--   and over land (completely dry column) where kSurfC = Nr+1
C-----------------------------------------------------------------------

      DO bj = myByLo(myThid), myByHi(myThid)
       DO bi = myBxLo(myThid), myBxHi(myThid)
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
          IF ( kSurfC(i,j,bi,bj).LE.Nr .AND.
     &         Ro_surf(i,j,bi,bj).LT.rF(1) ) THEN
            kTopC(i,j,bi,bj) = kSurfC(i,j,bi,bj)
          ELSE
            kTopC(i,j,bi,bj) = 0
          ENDIF
          shelficeMassInit   (i,j,bi,bj) = 0. _d 0
          shelficeLoadAnomaly(i,j,bi,bj) = 0. _d 0
          shelfIceMassDynTendency(i,j,bi,bj) = 0. _d 0
         ENDDO
        ENDDO
       ENDDO
      ENDDO

#ifdef ALLOW_SHIFWFLX_CONTROL
C     maskSHI is a hack to play along with the general ctrl-package
C     infrastructure, where only the k=1 layer of a 3D mask is used
C     for 2D fields. We cannot use maskInC instead, because routines
C     like ctrl_get_gen and ctrl_set_unpack_xy require 3D masks.
      DO bj = myByLo(myThid), myByHi(myThid)
       DO bi = myBxLo(myThid), myBxHi(myThid)
        DO k=1,Nr
         DO j=1-OLy,sNy+OLy
          DO i=1-OLx,sNx+OLx
           maskSHI(i,j,k,bi,bj) = 0. _d 0
          ENDDO
         ENDDO
        ENDDO
        DO k=1,Nr
         DO j=1-OLy,sNy+OLy
          DO i=1-OLx,sNx+OLx
           IF ( Ro_surf(i,j,bi,bj).LT.rF(1)
     &          .AND. hFacC(i,j,k,bi,bj).NE.zeroRS ) THEN
            maskSHI(i,j,k,bi,bj) = 1. _d 0
            maskSHI(i,j,1,bi,bj) = 1. _d 0
           ENDIF
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDDO
#endif /* ALLOW_SHIFWFLX_CONTROL */

#ifdef ALLOW_COST
#if  (defined (ALLOW_SHIFWFLX_COST_CONTRIBUTION) && \
      defined (ALLOW_SHIFWFLX_CONTROL))
      IF ( shifwflx_errfile .NE. ' ' ) THEN
       CALL READ_REC_XY_RL( shifwflx_errfile, wshifwflx, 1, 0, myThid )
      ENDIF

      DO bj = myByLo(myThid), myByHi(myThid)
       DO bi = myBxLo(myThid), myBxHi(myThid)
        DO j = 1-OLy, sNy+OLy
         DO i = 1-OLx, sNx+OLx
c--   Test for missing values.
          IF (wshifwflx(i,j,bi,bj) .LT. -9900.) THEN
           wshifwflx(i,j,bi,bj) = 0. _d 0
          ENDIF
c--   use weight as mask
          wshifwflx(i,j,bi,bj) =
     &         max(wshifwflx(i,j,bi,bj),wshifwflx0)
     &         *maskSHI(i,j,1,bi,bj)
          IF (wshifwflx(i,j,bi,bj) .NE. 0.) THEN
           wshifwflx(i,j,bi,bj) =
     &          1./wshifwflx(i,j,bi,bj)/wshifwflx(i,j,bi,bj)
          ENDIF
         ENDDO
        ENDDO
       ENDDO
      ENDDO
      CALL ACTIVE_WRITE_XY_LOC( 'wshifwflx', wshifwflx,
     &                          1, 0, myThid, dummy )
#endif /* ALLOW_SHIFWFLX_COST_CONTRIBUTION and ALLOW_SHIFWFLX_CONTROL */
#endif /* ALLOW_COST */

      IF ( SHELFICEloadAnomalyFile .NE. ' ' ) THEN
       CALL READ_FLD_XY_RL( SHELFICEloadAnomalyFile, ' ',
     &                      shelficeLoadAnomaly, 0, myThid )
      ENDIF
      IF ( SHELFICEmassFile.NE.' ' ) THEN
       CALL READ_FLD_XY_RL( SHELFICEmassFile, ' ',
     &                      shelficeMassInit, 0, myThid )
      ELSE
       DO bj = myByLo(myThid), myByHi(myThid)
        DO bi = myBxLo(myThid), myBxHi(myThid)
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
           shelficeMassInit(i,j,bi,bj) =
     &         shelficeLoadAnomaly(i,j,bi,bj)*recip_gravity
     &       - rhoConst*Ro_surf(i,j,bi,bj)
          ENDDO
         ENDDO
        ENDDO
       ENDDO
      ENDIF
      _EXCH_XY_RL( shelficeMassInit, myThid )
      CALL WRITE_FLD_XY_RL ( 'shelficemassinit', ' ',
     &                       shelficeMassInit, 0, myThid )

c     IF ( SHELFICEloadAnomalyFile .EQ. ' ' ) THEN
C-   In case we need shelficeLoadAnomaly in phi0surf for initial pressure
C    calculation (if using selectP_inEOS_Zc=2 or 3)
c     wykang: def
      rho_out=1000.
      rho_bulk=gravity/((6.67 _d -11) *16.*atan(1.)/3.*rSphere)
       DO bj = myByLo(myThid), myByHi(myThid)
        DO bi = myBxLo(myThid), myBxHi(myThid)
         DO j = 1-OLy, sNy+OLy
          DO i = 1-OLx, sNx+OLx
c          wykang: gravity variation with r
           H_ice=shelficeMassInit(I,J,bi,bj)/rhoShelfIce
c           p_correction=((1.-rho_out/rho_bulk)/(1.-H_ice/rSphere)
c     &              +rho_out/rho_bulk*(1.-H_ice/rSphere/2.))
           p_correction=1.
           shelficeLoadAnomaly(i,j,bi,bj) = gravity
     &      *(shelficeMass(i,j,bi,bj)*p_correction 
     &      + rhoConst*Ro_surf(i,j,bi,bj))
c     &      *(shelficeMassInit(i,j,bi,bj)+rhoConst*Ro_surf(i,j,bi,bj))
          ENDDO
         ENDDO
        ENDDO
       ENDDO
c     ELSE
c      _EXCH_XY_RS( shelficeLoadAnomaly, myThid )
c     ENDIF
      IF ( debugLevel.GE.debLevC ) THEN
       CALL WRITE_FLD_XY_RL( 'SHICE_pLoadAnom', ' ',
     I                       shelficeLoadAnomaly, -1, myThid )
      ENDIF

      IF ( SHELFICEMassStepping .AND.
     &     SHELFICEMassDynTendFile .NE. ' ' ) THEN
       CALL READ_FLD_XY_RS( SHELFICEMassDynTendFile, ' ',
     &                      shelfIceMassDynTendency, 0, myThid )
      ENDIF

#ifdef ALLOW_DIAGNOSTICS
      IF ( useDiagnostics ) THEN
       diagName  = 'SHIfwFlx'
       diagTitle = 'Ice shelf fresh water flux (positive upward)'
       diagUnits = 'kg/m^2/s        '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

       diagName  = 'SHIhtFlx'
       diagTitle = 'Ice shelf heat flux  (positive upward)'
       diagUnits = 'W/m^2           '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

       diagName  = 'SHI_TauX'
       diagTitle =
     &     'Ice shelf bottom stress, zonal  comp., >0 increases uVel'
       diagUnits = 'N/m^2           '
       diagCode  = 'UU      L1      '
       diagMate  = diagNum + 2
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, diagMate, myThid )

       diagName  = 'SHI_TauY'
       diagTitle =
     &     'Ice shelf bottom stress, merid. comp., >0 increases vVel'
       diagUnits = 'N/m^2           '
       diagCode  = 'VV      L1      '
       diagMate  = diagNum
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, diagMate, myThid )

       diagName  = 'SHIForcT'
       diagTitle = 'Ice shelf forcing for theta, >0 increases theta'
       diagUnits = 'W/m^2           '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

       diagName  = 'SHIForcS'
       diagTitle = 'Ice shelf forcing for salt, >0 increases salt'
       diagUnits = 'g/m^2/s         '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

       diagName  = 'SHIgammT'
       diagTitle = 'Ice shelf exchange coefficient for theta'
       diagUnits = 'm/s             '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

       diagName  = 'SHIgammS'
       diagTitle = 'Ice shelf exchange coefficient for salt'
       diagUnits = 'm/s             '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

       diagName  = 'SHIuStar'
       diagTitle = 'Friction velocity at bottom of ice shelf'
       diagUnits = 'm/s             '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

       diagName  = 'SHI_mass'
       diagTitle = 'dynamic ice shelf mass for surface load anomaly'
       diagUnits = 'kg/m^2          '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

       diagName  = 'SHI_flow'
       diagTitle = 'dynamic ice mass flow induced tendency'
       diagUnits = 'kg/m^2/s        '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )

#ifdef ALLOW_SHELFICE_REMESHING
       diagName  = 'SHIRshel'
       diagTitle = 'depth of shelfice'
       diagUnits = 'm               '
       diagCode  = 'SM      L1      '
       CALL DIAGNOSTICS_ADDTOLIST( diagNum,
     I      diagName, diagCode, diagUnits, diagTitle, 0, myThid )
#endif

      ENDIF
#endif /* ALLOW_DIAGNOSTICS */
#endif /* ALLOW_SHELFICE */

      RETURN
      END
