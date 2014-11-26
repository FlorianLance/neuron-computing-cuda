use strict;
use warnings;
use Cwd;

my $PathsToAdd = ";";
$PathsToAdd = "D:/_software/gnuplot/bin;";
$ENV{PATH} = $PathsToAdd . $ENV{PATH};

system("gnuplot ../dist/plotRes2.gplt");

