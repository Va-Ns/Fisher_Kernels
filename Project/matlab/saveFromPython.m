function saveFromPython(path, varargin)
arguments
    path (:, 1) string
end
arguments(Repeating)
    varargin
end
save(path + "/python-data.mat", "varargin");
end