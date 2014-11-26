my $ARG1 = "";
my $ARG2 = "";
my $ARG3 = "";
my $ARG4 = "";
my $ARG5 = "";
my $ARG6 = "";
my $ARG7 = "";
if(@ARGV > 0)
{
  $ARG1 = @ARGV[0];
}
if(@ARGV > 1)
{
  $ARG2 = @ARGV[1];
}
if(@ARGV > 2)
{
  $ARG3 = @ARGV[2];
}
if(@ARGV > 3)
{
  $ARG4 = @ARGV[3];
}
if(@ARGV > 4)
{
  $ARG5 = @ARGV[4];
}
if(@ARGV > 5)
{
  $ARG6 = @ARGV[5];
}
if(@ARGV > 6)
{
  $ARG7 = @ARGV[6];
}
system("\"C:\Windows/system32/xcopy\" /q /e /y /c  \"D:/neuron-computing-cuda/scripts/../reservoir-cuda-config\" \"D:/neuron-computing-cuda/scripts/../dist/data\"");
chdir "D:/neuron-computing-cuda/scripts/";
{
   local @ARGV = ("amd64");
   require "win-init_env_command.pl";
}
chdir "../dist/bin";
system("reservoir-test-x64.exe" . " " . $ARG1 . " " . $ARG2 . " " . $ARG3 . " " . $ARG4 . " " . $ARG5 . " " . $ARG6 . " " . $ARG7);
chdir "..";
