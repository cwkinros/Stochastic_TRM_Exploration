tests = 'hello my name is colleen';

% task is to get each word as we go

[~,len] = size(tests);

i = 1;

while i <= len
    start = i;
    while i <= len && tests(i) ~= ' '
        i = i + 1;
    end
    test = tests(start:i-1);
    disp(test);
    i = i + 1;
end