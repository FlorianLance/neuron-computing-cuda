use strict;
use warnings;
use Cwd;

my $CurrentDirectory = cwd();

package Env;

my $ARGNb = 0;
my $ARG = "";

if(@ARGV > 0)
{
    $ARGNb = @ARGV;
    $ARG = $ARGV[0];
}

####################################################################################### UTILITY DIRECTORIES
our $PBase         = $CurrentDirectory . "/../";
our $PDist        = $PBase . "dist/";
$ENV{P_DIST}      = $PDist;
our $PScripts     = $CurrentDirectory . "/";
our $PDistLIB     = $PDist . "lib/";
our $PThird_party  = $PBase . "../swooz/swooz-3rdparty/";
our $PLocalThird_party = $PBase . "3rdParty/";


####################################################################################### CHECK CUDA PATH


if (length($ENV{CUDA_PATH}) > 1)
{
    $ENV{CUDA_FOUND} = "yes";
}
else
{
    $ENV{CUDA_FOUND} = "no";
}

####################################################################################### SET PROJECTS TO BUILD
our $Reservoir   = "reservoir";

# add projects to build here
my @PbuildOrder            = ($Reservoir);

sub buildOrder {
    return @PbuildOrder;
}

# executables scripts to be generated for each project
my @ExecScriptParams = (# reservoir
                        "reservoir-x64",   "amd64",  "reservoir-x64.exe",
                        "reservoir-interface-x64",   "amd64",  "reservoir-interface-x64.exe",
                        "reservoir-yarp-x64", "amd64", "reservoir-yarp-x64.exe",
                        "test-yarp-x64", "amd64", "test-yarp-x64.exe"
);

sub executablesNumber {
    my $size = $#ExecScriptParams+1;
    return ($size / 3);
};

sub commandFileName {
    my ($num) = @_;
    return $ExecScriptParams[($num * 3)];
};

sub archExe {
    my ($num) = @_;
    return $ExecScriptParams[(1 + $num * 3)];
};

sub exeFileName {
    my ($num) = @_;
    return $ExecScriptParams[(2 +$num * 3)];
};

####################################################################################### SET 3RD PARTY SOFTWARE PATH
##################### PYTHON
$ENV{THIRD_PARTY_PYTHON}   = $PLocalThird_party . "python/";
$ENV{THIRD_PARTY_PYTHON_64}= $PLocalThird_party . "python/amd64/";
##################### TOBII
$ENV{THIRD_PARTY_TOBII}   = $PThird_party . "tobii/";
##################### FACELAB
$ENV{THIRD_PARTY_FACELAB}   = $PThird_party . "facelab/coredata-3.2.1.65799_vc10/";
##################### YARP
$ENV{THIRD_PARTY_YARP}   = $PLocalThird_party . "yarp/yarp-2.3.60/";
$ENV{THIRD_PARTY_YARP_64}= $PLocalThird_party . "yarp/amd64/yarp-2.3.60/";
##################### ACE
$ENV{THIRD_PARTY_ACE}   = $PLocalThird_party . "ace/ace-6.1.0/";
$ENV{THIRD_PARTY_ACE_64}= $PLocalThird_party . "ace/amd64/ace-6.1.0/";
##################### OPENNI
$ENV{THIRD_PARTY_OPENNI}    = $PThird_party . "openni/openni-1.5.4.0/";
$ENV{THIRD_PARTY_OPENNI2_64}= $PThird_party . "openni/amd64/openni-2.1.0.4-64/";
##################### OPENCV
$ENV{THIRD_PARTY_OPENCV}   = $PThird_party . "opencv/opencv-2.49/";
$ENV{THIRD_PARTY_OPENCV_64}= $PThird_party . "opencv/amd64/opencv-2.49/";
##################### FREEGLUT
$ENV{THIRD_PARTY_FREEGLUT}   = $PThird_party . "freeglut/";
##################### CUBLAS
$ENV{THIRD_PARTY_CUBLAS_64}  = $PLocalThird_party . "cublas/amd64/";
##################### QT
$ENV{THIRD_PARTY_QT}    = $PThird_party . "Qt/4.8.3/";
$ENV{THIRD_PARTY_QT_64} = $PThird_party . "Qt/amd64/4.8.3/";
#$ENV{THIRD_PARTY_QT}    = $PLocalThird_party . "qt/4.8.3/";
#$ENV{THIRD_PARTY_QT} = $PLocalThird_party . "qt/5.1.0/";
#$ENV{THIRD_PARTY_QT_64} = $PLocalThird_party . "qt/amd64/5.1.0/";
##################### BOOST
$ENV{THIRD_PARTY_BOOST}  = $PLocalThird_party . "boost/";
$ENV{THIRD_PARTY_BOOST_64} = $PLocalThird_party . "boost/amd64/";
##################### EIGEN
$ENV{THIRD_PARTY_EIGEN}   = $PThird_party . "Eigen/";
##################### VTK
$ENV{THIRD_PARTY_VTK}   = $PThird_party . "VTK/";
##################### FLANN
$ENV{THIRD_PARTY_FLANN}   = $PThird_party . "FLANN/";
##################### CUDA
#$ENV{THIRD_PARTY_CUDA}   = $PThird_party . "CUDA/";
#$ENV{THIRD_PARTY_CUDA_64}   = $PThird_party . "CUDA/amd64/";
##################### CULA
$ENV{THIRD_PARTY_CULA}   = $PThird_party . "CULA/";
$ENV{THIRD_PARTY_CULA_64}= $PThird_party . "CULA/amd64/";
##################### CLAPACK
$ENV{THIRD_PARTY_CLAPACK}   = $PThird_party . "CLAPACK/";
$ENV{THIRD_PARTY_CLAPACK_64}= $PThird_party . "CLAPACK/amd64/";
##################### GSL
$ENV{THIRD_PARTY_GSL}   = $PThird_party . "GSL/";
##################### GLM
$ENV{THIRD_PARTY_GLM}   = $PThird_party . "glm/glm-0.9.4.2/";
##################### POLHEMUS
$ENV{THIRD_PARTY_POLHEMUS}   = $PThird_party . "Polhemus/";
##################### NAOQI
$ENV{THIRD_PARTY_NAOQI}   = $PThird_party . "naoqi/";
##################### LEAP
$ENV{THIRD_PARTY_LEAP}   = $PThird_party . "leap/";
##################### ICUB
$ENV{THIRD_PARTY_ICUB}   = $PThird_party . "icub/";

$ENV{THIRD_PARTY_EIGEN}  = $PLocalThird_party . "Eigen/";
$ENV{THIRD_PARTY_ARMADILLO}  = $PLocalThird_party . "armadillo/include/";


our $CurrentPath = $ENV{PATH};

my $PathsToAdd = ";";


#$PathsToAdd = $ENV{THIRD_PARTY_QT} . "bin/;" . $PathsToAdd;
$PathsToAdd = $ENV{THIRD_PARTY_QT_64} . "bin/;" . $PathsToAdd;
$PathsToAdd = $ENV{CUDA_PATH} . "/bin/;" . $PathsToAdd;


$ENV{THIRD_PARTY_CUDA}   = $ENV{CUDA_PATH} . "/lib/Win32";
$ENV{THIRD_PARTY_CUDA_64}= $ENV{CUDA_PATH} . "/lib/x64";


####################################################################################### x86 PATHS
if($ARG eq "x86")
{
    $PathsToAdd = $ENV{THIRD_PARTY_TOBII} . "win32/binaries/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_FACELAB} . "coredata/libshared/win32_ia32sse2_vc90/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_YARP} . "bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_ACE} . "bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_OPENNI} . "Bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_OPENCV} . "build/bin/Release;" . $PathsToAdd;
    $PathsToAdd = $ENV{THIRD_PARTY_OPENCV} . "build/bin/Debug;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_QT} . "bin/;" . $PathsToAdd;
    $PathsToAdd = $ENV{THIRD_PARTY_QT} . "lib/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_BOOST} . "lib/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_FLANN} . "lib/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_CULA} . "bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_CLAPACK} . "lib/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_GSL} . "lib/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_POLHEMUS} . "lib/Win32/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_FREEGLUT} . "bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_NAOQI} . "bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_LEAP} . "lib/;" . $PathsToAdd;
}

####################################################################################### amd64 PATHS
if($ARG eq "amd64")
{
    $PathsToAdd = $ENV{THIRD_PARTY_OPENNI2_64} . "Bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_OPENCV_64} . "build/bin/Release/;" . $PathsToAdd;
    $PathsToAdd = $ENV{THIRD_PARTY_OPENCV_64} . "build/bin/Debug/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_BOOST_64} . "lib/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_ACE_64} . "bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_CULA_64} . "lib/;" . $PathsToAdd;
    $PathsToAdd = $ENV{THIRD_PARTY_CULA_64} . "bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_QT_64} . "lib/;" . $PathsToAdd;
    $PathsToAdd = $ENV{THIRD_PARTY_QT_64} . "bin/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_CLAPACK_64} . "lib/;" . $PathsToAdd;

    $PathsToAdd = $ENV{THIRD_PARTY_CUBLAS_64} . "bin/;" . $PathsToAdd;

}

####################################################################################### UPDATE GLOBAL PATH


$ENV{PATH} = $PathsToAdd . $ENV{PATH};
