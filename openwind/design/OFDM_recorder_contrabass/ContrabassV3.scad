//boreRadius = 22;
//wallThickness = 6;

//$fn = 50;

//path = [[-50,150,50],[0,150,50],[50,150,50],[50,100,50],[50,100,100],[100,100,100],[150,100,100],[150,100,50],[150,150,50],[150,150,100],[150,150,150],[100,150,150],[50,150,150],[50,150,100],[100,150,100],[100,150,50],[100,100,50],[100,50,50],[150,50,50],[150,50,100],[150,50,150],[150,100,150],[100,100,150],[50,100,150],[50,50,150],[100,50,150],[100,50,100],[50,50,100],[50,50,50],[50,50,0]];

//vents = [[50,150,125],[100,150,100],[100,100,50],[150,50,100],[150,50,150],[50,75,150]];

boreRadius = 28;
wallThickness = 6;

path = [[62,124,62],[62,124,124],[62,124,186],[62,62,186],[62,62,124],[124,62,124],[124,124,124],[186,124,124],[186,186,124],[186,186,186],[186,124,186],[124,124,186],[124,62,186],[124,62,248],[62,62,248],[62,124,248],[124,124,248],[124,186,248],[124,186,186],[124,186,124],[124,186,62],[186,186,62],[186,124,62],[124,124,62],[124,62,62],[186,62,62],[186,62,124],[186,62,186],[186,62,248],[186,124,248],[186,186,248]];

vents = [[124,62,186],[62,62,248],[124-5,124+5,248],[124,186,124],[124,186,62],[124+5,124-5,62]];

module wedge(h, r, a, $fn=20)
{
    th=(a%360)/2;
    difference()
    {
        cylinder(h=h,r=r,center=true, $fn=$fn);
        if(th<90)
        {
            for(n=[-1,1])rotate(-th*n)translate([(r+0.5)*n,0,0])
                cube(size=[r*2+1,r*2+1,h+1],center=true);
        }
        else
        {
            intersection()
            {
                rotate(-th)translate([(r+0.5),(r+0.5),0])
                    cube(size=[r*2+1,r*2+1,h+1],center=true);
                rotate(th)translate([-(r+0.5),(r+0.5),0])
                    cube(size=[r*2+1,r*2+1,h+1],center=true);
            }
        }
    }
}

module torus(r1=1, r2=2, angle=360, endstops=0, $fn=20){
    if(angle < 360){
        intersection(){
            rotate_extrude(convexity=10, $fn=$fn)
            translate([r2, 0, 0])
            circle(r=r1, $fn=$fn);
            
            color("blue")
            wedge(h=r1*3, r=r2*2, a=angle);
        }
    }else{
        rotate_extrude(convexity=10, $fn=$fn)
        translate([r2, 0, 0])
        circle(r=r1, $fn=$fn);
    }
    
    if(endstops && angle < 360){
        rotate([0,0,angle/2])
        translate([0,r2,0])
        sphere(r=r1);
        
        rotate([0,0,-angle/2])
        translate([0,r2,0])
        sphere(r=r1);
    }
}

module rounded_cylinder(d=1, h=1, r=0.1, center=true, $fn=50){
    translate([0,0,(center==true)?-h/2:0]){
        union(){
            // bottom edge
            translate([0,0,r])torus(r1=r, r2=(d-r*2)/2, $fn=$fn);
            // top edge
            translate([0,0,h-r])torus(r1=r, r2=(d-r*2)/2, $fn=$fn);
            // main cylinder outer
            translate([0,0,r])cylinder(d=d, h=h-r*2, center=false, $fn=$fn);
            // main cylinder inner
            translate([0,0,0])cylinder(d=d-r*2, h=h, center=false, $fn=$fn);
        }
    }
}

module elbow(){
fullRad = boreRadius + wallThickness;
hgt = 2*fullRad;
bInit = 135;  
intersection(){
    translate([-fullRad,-fullRad,-fullRad]) cube([2*fullRad,2*fullRad,2*fullRad]);
    difference(){
        translate([fullRad,fullRad,0]) rotate([0,0,bInit]) torus(r1=fullRad-1,r2=fullRad+0,angle=90, endstops=1);
        translate([fullRad,fullRad,0]) rotate([0,0,bInit]) torus(r1=boreRadius,r2=fullRad,angle=90, endstops=1);};
}
}

module bores(){
fullRad = boreRadius + wallThickness;
hgt = 2*fullRad;
bInit = 135;    
for (a = [ 1 : len(path) - 2 ]) 
{
    preDir = (path[a-1]-path[a])/(hgt-6);//50;
    //echo(preDir);
    postDir = (path[a+1]-path[a])/(hgt-6);//50;
    //echo(postDir);
    asdf = postDir[0]>0;
    //echo(asdf);
    //Generate and place the straight bores:
    if ((abs(preDir[0])==1)&&(abs(postDir[0])==1))
        difference(){translate(path[a]) rotate([0,90,0]) translate([0,0,-hgt/2]) cylinder(h=hgt,r=fullRad-1);
    translate(path[a]) rotate([0,90,0]) translate([0,0,-hgt/2]) cylinder(h=hgt,r=boreRadius);}
    if ((abs(preDir[1])==1)&&(abs(postDir[1])==1))
        difference(){translate(path[a]) rotate([90,0,0]) translate([0,0,-hgt/2]) cylinder(h=hgt,r=fullRad-1);
    translate(path[a]) rotate([90,0,0]) translate([0,0,-hgt/2]) cylinder(h=hgt,r=boreRadius);}
    if ((abs(preDir[2])==1)&&(abs(postDir[2])==1))
        difference(){translate(path[a]) rotate([0,0,0]) translate([0,0,-hgt/2]) cylinder(h=hgt,r=fullRad-1);
    translate(path[a]) rotate([0,0,0]) translate([0,0,-hgt/2]) cylinder(h=hgt,r=boreRadius);}
    
    //Generate and place the right angle bores:
    totDir = preDir + postDir;
    //echo(totDir);
    //if ((totDir[0]==1)&&(totDir[1]==1))
    //    difference(){translate(path[a]) rotate([0,0,0]) translate([fullRad,fullRad,0]) rotate([0,0,bInit]) torus(r1=fullRad+0,r2=fullRad+0,angle=90, endstops=1);
    //translate(path[a]) rotate([0,0,0]) translate([fullRad,fullRad,0]) rotate([0,0,bInit]) torus(r1=boreRadius,r2=fullRad,angle=90, endstops=1);};
    
        if ((totDir[0]==1)&&(totDir[1]==1))
        translate(path[a]) rotate([0,0,0]) elbow();
        
        if ((totDir[0]==-1)&&(totDir[1]==1))
        translate(path[a]) rotate([0,0,90]) elbow();
        
        if ((totDir[0]==-1)&&(totDir[1]==-1))
        translate(path[a]) rotate([0,0,180]) elbow();
        
        if ((totDir[0]==1)&&(totDir[1]==-1))
        translate(path[a]) rotate([0,0,-90]) elbow();
        
        
        if ((totDir[1]==1)&&(totDir[2]==1))
        translate(path[a]) rotate([0,-90,0]) elbow();
        
        if ((totDir[1]==-1)&&(totDir[2]==1))
        translate(path[a]) rotate([90,0,-90]) elbow();
        
        if ((totDir[1]==-1)&&(totDir[2]==-1))
        translate(path[a]) rotate([-90,0,-90]) elbow();
        
        if ((totDir[1]==1)&&(totDir[2]==-1))
        translate(path[a]) rotate([0,90,0]) elbow();
        
        
        if ((totDir[0]==1)&&(totDir[2]==1))
        translate(path[a]) rotate([90,0,0]) elbow();
        
        if ((totDir[0]==-1)&&(totDir[2]==1))
        translate(path[a]) rotate([90,0,180]) elbow();
        
        if ((totDir[0]==-1)&&(totDir[2]==-1))
        translate(path[a]) rotate([-90,0,180]) elbow();
        
        if ((totDir[0]==1)&&(totDir[2]==-1))
        translate(path[a]) rotate([-90,0,0]) elbow();    
}
}

module vents(){
    venRad = 6;
    //angs = [[-90,0,70],[-90,0,-25],[0,140,0],[90,0,0],[40,0,50],[0,-30,0]];
    angs = [[90+30,0,0],[90,0,-40],[20,20,0],[-90,0,0],[-90-30,0,30],[0-20,180+20,0]];
    for (a = [ 0 : len(vents) - 1 ]) 
{
    translate(vents[a]) rotate(angs[a]) translate([0,0,-venRad/2]) cylinder(h=60,r=venRad);
}
}

module fullAirPath(){
difference(){
bores();
vents();
}
}

module enclosure(){
difference() {
translate([25,25,25]) cube([150,150,150]);
translate([25,75,25]) cube([50,100,50]);    
}
}

//elbow();
//enclosure();
//difference(){
//    enclosure();
translate([-32,220,282]) rotate([180,0,0]) fullAirPath();
//}
//difference(){
//translate([50, 100, 50]) rotate([0, -180, 0]) import("/Users/rkmoore/Desktop/bass.stl");
//translate([25,25,25]) cube([50,50,50]);
//}