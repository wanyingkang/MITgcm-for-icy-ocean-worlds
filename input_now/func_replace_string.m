function [] = func_replace_string(InputFile, SearchString, ReplaceString,wholeline)
%%change data [e.g. initial conditions] in model file
% InputFile - string
% SearchString - string
% ReplaceString - string
% read whole model file data into cell array
if nargin<4
    wholeline=1;
    % search pattern and replace the wholeline by default
end

fid = fopen(InputFile);
data = textscan(fid, '%s', 'Delimiter', '\n', 'CollectOutput', true);
fclose(fid);
% modify the cell array
% find the position where changes need to be applied and insert new data
found=0;
for I = 1:length(data{1})
    if wholeline
        tf = contains(data{1}{I}, SearchString);
    else
        tf = strcmp(data{1}{I}, SearchString); % search for this string in the array
    end
    if tf == 1
        found=1;
        data{1}{I} = ReplaceString; % replace with this string
    end
end
% write the modified cell array into the text file
%system(['mv -f ',InputFile,' ',InputFile,'_back'])
fid = fopen(InputFile, 'w');
for I = 1:length(data{1})
    if ~found && strcmp(char(data{1}{I}),'&')
        fprintf(fid, '%s\n', ReplaceString);
    end
    fprintf(fid, '%s\n', char(data{1}{I}));
end
fclose(fid);
