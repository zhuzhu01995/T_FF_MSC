function x0 = glu(la,a,x)
    %The expression of the fractional threshold function 
    f = acos(-1+(27*la*a*a)/(4.0*(1+a*abs(x))^3.0));
    x0 = sign(x)*(((1+a*abs(x))*(1+2*cos(f/3-pi/3))-3)/(3.0*a));
end