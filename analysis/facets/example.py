#!/usr/bin/env python
import facets as fa
import exchange

ncs = 510
dims = 6*[510, 510]

exch = exchange.cs()

# read grid info
grid = fa.MITGrid('tile{:03d}.mitgrid', dims=dims)
xg = grid.xg
yg = grid.yg
Ac = grid.ac
print(xg.shapes)
print(Ac.shapes)

# read from mds and add halo
v = fa.frommds('RAC', dims=dims, halo=1)
exch(v, halo=1)
print(v.shapes)

dvdx = v[...,1:] - v[...,:-1]
print(dvdx.shapes)

# access cube facets
f1 = v.F[0]
print(f1.shape)

# map to global array (removing halo)
a = v.toglobal(halo=1)
print(a.shape)

import facetplot as fp

fp.pcolormeshes(xg, yg, v[...,1:-1,1:-1])
