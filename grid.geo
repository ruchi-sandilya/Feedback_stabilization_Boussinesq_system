// Gmsh code for generating locally refined mesh
ho = 1.0/100.0;
hi = 1.0/100.0;
hh = 1.0/50.0;
hc = 1.0/50.0;

n1 = 40; r1 = 0.4;
n2 = 40; r2 = 0.4;  // heat flux
n3 = 50; r3 = 0.5;
n4 = 50; r4 = 0.1;
n5 = 40; r5 = 0.4;  // inlet
n6 = 20; r6 = 0.4;
n7 = 50; r7 = 0.1;  // top
n8 = 40; r8 = 0.1;
n9 = 40; r9 = 0.4;  // outlet
n10= 20; r10= 0.3;

x1 = 0.4;
x2 = 0.6;
y1 = 0.1;
y2 = 0.4;
y3 = 0.7;
y4 = 0.9;

Point(1) = {0, 0, 0, hc};
Point(2) = {x1, 0, 0, hh};
Point(3) = {x2, 0, 0, hh};
Point(4) = {1, 0, 0, hc};
Point(5) = {1, y3, 0, hi};
Point(6) = {1, y4, 0, hi};
Point(7) = {1, 1, 0, hc};
Point(8) = {0, 1, 0, hc};
Point(9) = {0, y2, 0, ho};
Point(10) = {0, y1, 0, ho}; 

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,1};

Line Loop(1) = {1,2,3,4,5,6,7,8,9,10};
Plane Surface(1) = {1};


Transfinite Line{1} = n1 Using Bump r1;
Transfinite Line{2} = n2 Using Bump r2;
Transfinite Line{3} = n3 Using Bump r3;
Transfinite Line{4} = n4 Using Bump r4;
Transfinite Line{5} = n5 Using Bump r5;
Transfinite Line{6} = n6 Using Bump r6;
Transfinite Line{7} = n7 Using Bump r7;
Transfinite Line{8} = n8 Using Bump r8;
Transfinite Line{9} = n9 Using Bump r9;
Transfinite Line{10} = n10 Using Bump r10;

