% makes grid figure

x = 1:29;
to_y = ones(29,1);
figure;
for i = 1:29
    y = to_y*i;
    line(x,y);
    line(y,x);
end


line(x,5*to_y,'LineWidth', 5.0);
line(x,9*to_y,'LineWidth', 5.0);
line(x,11*to_y,'LineWidth', 5.0);
line(x,13*to_y,'LineWidth', 5.0);
line(x,15*to_y,'LineWidth', 5.0);
line(x,17*to_y,'LineWidth', 5.0);
line(x,19*to_y,'LineWidth', 5.0);
line(x,21*to_y,'LineWidth', 5.0);
line(x,25*to_y,'LineWidth', 5.0);
line(5*to_y,x,'LineWidth', 5.0);
line(9*to_y,x,'LineWidth', 5.0);
line(11*to_y,x,'LineWidth', 5.0);
line(13*to_y,x,'LineWidth', 5.0);
line(15*to_y,x,'LineWidth', 5.0);
line(17*to_y,x,'LineWidth', 5.0);
line(19*to_y,x,'LineWidth', 5.0);
line(21*to_y,x,'LineWidth', 5.0);
line(25*to_y,x,'LineWidth', 5.0);