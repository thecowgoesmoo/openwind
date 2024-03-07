function instr = BoreOpt(varargin)

%This function will take a bore definition, set of constraints, and attempt
%to produce a bore path that will mee tthe constraints.
%
%Richard Moore
%March 7, 2024
%are.kay.more@ymail.com
%https://github.com/thecowgoesmoo
%Creative Commons Attribution Share Alike 4.0

switch nargin
    case 0
        %Run using a default demo values for bore and constraints
        %Bore representation:
        bore.rad = 22;
        bore.wall = 3;
        bore.len = 1300;
        bore.vent = [600; 700; 800; 900; 1000; 1100; inf];
        
        constr.printVol = [220; 220; 250];
        constr.maxVentDists = [0 100 100 inf inf inf;...
            100 0 100 inf inf inf;...
            100 100 0 inf inf inf;...
            inf inf inf 0 100 100;...
            inf inf inf 100 0 100;...
            inf inf inf 100 100 0];
    case 1
        bore = varargin{1};
        constr.printVol = [220; 220; 250];
        constr.maxVentDists = [0 100 100 inf inf inf;...
            100 0 100 inf inf inf;...
            100 100 0 inf inf inf;...
            inf inf inf 0 100 100;...
            inf inf inf 100 0 100;...
            inf inf inf 100 100 0];
    case 2
        bore = varargin{1};
        constr = varargin{2};
end

%to populate boxes:
%   1. A cylinder in one of 3 possible orientations
%   2. An elbow in one of 12 possible orientations
%   3. A blank space
%16 possible options for each cube
%For 50 mm cubes in a 200 mm cube print volume, that means 64 locations.
%So, 16^64 possible configs.  Most of those are discontinuous nonsense,
%though.

%okay.  A more constrained search would be to start at a random edge
%location and snake randomly through available cubes.  Then check to see if
%the vent proximity constraints are met.

%Maybe constrain to a single edge for starting.  Symmetry makes solutions
%from the 12 different edges identical.

%Okay, I'll try doing pretty much the old Windows pipes screensaver thing.
%Just randomly noodle around a pipe through discretized space at right
%angles.  The vent holes will land where they land and if the full bore
%length can be achieved without hitting itself or leaving the print volume,
%a check will be run to see if the max distance limits matrix is
%maintained.  

%It is not a smart search, but it is an automated search. 

fullBoreD = bore.rad*2+bore.wall*2;

bDims = floor(constr.printVol./fullBoreD)';
disLen = 0;

distFlag = 0;

%Keep searching until the vent distance constraints are met:

while ~distFlag
    disLen = fullBoreD;
    
    path = [];
    
    %Keep randomly noodling bore paths until the full bore length is
    %achieved:
    while disLen<bore.len
        board = zeros(bDims);
        strLen = fullBoreD;
        bndLen = fullBoreD*pi/2;
        disLen = strLen;
        disPos = [1 1 1];
        fCtr = 0;
        ventPos = [];
        path = [];
        ventInd = 1;
        
        %Keep rolling the die until the bore can be extended to an
        %available grid cube:
        while (disLen<bore.len)&&(fCtr<20)
            dirDie = ceil(6*rand);
            canPos = disPos;
            
            %Pick a random next direction:
            switch dirDie
                case 1
                    canPos(1) = canPos(1)+1;
                case 2
                    canPos(1) = canPos(1)-1;
                case 3
                    canPos(2) = canPos(2)+1;
                case 4
                    canPos(2) = canPos(2)-1;
                case 5
                    canPos(3) = canPos(3)+1;
                case 6
                    canPos(3) = canPos(3)-1;
            end
            fCtr = fCtr + 1;
            
            %Check to see if canPos is available
            if (prod(canPos>0))&&(prod(canPos<=bDims))
                if board(canPos(1),canPos(2),canPos(3))==0 %(board(canPos)==0)
                    disLen = disLen + strLen;
                    board(canPos(1),canPos(2),canPos(3)) = disLen;
                    disPos = canPos;
                    fCtr = 0;
                    path = [path; disPos];
                    if disLen>bore.vent(ventInd)
                        ventPos = [ventPos; disPos];
                        ventInd = ventInd + 1;
                    end
                end
            end
        end
        %disp(['Final disLen = ' num2str(disLen)]);
    end
    
    %disp(board);
    
    %Check to see if vent location proximity constraints are met:
    numVents = size(constr.maxVentDists,1);
    for x = 1:numVents
        for y = 1:numVents
            %dists(x,y) = strLen.*rssq(ventPos(x,:)-ventPos(y,:));
        dists(x,y) = strLen.*(sum((ventPos(x,:)-ventPos(y,:)).^2)).^0.5;
        end
    end
    checkDists = dists<=constr.maxVentDists;
    %disp(checkDists);
    distFlag = prod(checkDists(:));
end

path = path.*fullBoreD;
ventPos = ventPos.*fullBoreD;

%Plot the path discovered during the search:
figure,plot3(path(:,1),path(:,2),path(:,3),'-b','LineWidth',3);
hold on;
plot3(ventPos(1:3,1),ventPos(1:3,2),ventPos(1:3,3),'o','MarkerSize',20,'LineWidth',3);
plot3(ventPos(4:6,1),ventPos(4:6,2),ventPos(4:6,3),'o','MarkerSize',20,'LineWidth',3);
plot3(fullBoreD,fullBoreD,fullBoreD,'.','MarkerSize',20,'LineWidth',3)
plot3(path(end,1),path(end,2),path(end,3),'*','MarkerSize',20,'LineWidth',3)
grid on;
axis equal;

instr.path = path;
instr.ventPos = ventPos;
instr.board = board;


