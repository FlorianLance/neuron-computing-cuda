
if(chdir "./dist/doc/Reservoir-cuda/html")
{
    system("index.html");
    1;
}

chdir "../../../..";
