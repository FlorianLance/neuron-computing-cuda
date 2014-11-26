RESERVOIR-CUDA - SCRIPTS
=========================

This README is only for Windows 7, others Windows have not been tested.


0. DEPENDENCIES
---------------

 * Perl is needed for launching the scripts, you can install this perl distribution : http://strawberryperl.com/

 * The scripts use the VS 2010 compiler to build the solution, if you don't have any version of Visual Studio you can install the [Microsoft SDK](http://www.microsoft.com/en-us/download/details.aspx?id=8279)

1. BUILD RESERVOIR-CUDA
-----------------------

Go to ./scripts and open a console.

	* win-build.pl builds the platform :
		* win-build 		   -> for the release x86 mode
		* win-build Debug          -> for the debug x86 mode 
		* win-build Release  	   -> for the release x86 mode
		* win-build Debug adm64    -> for the debug amd64 mode 
		* win-build Release adm64  -> for the release amd64 mode
	
This script creates the ../dist file tree :  

	* ../dist
		* ../dist/bin
		* ../dist/include
		* ../dist/lib
			* ../dist/lib/x86
				* ../dist/lib/x86/Debug
				* ../dist/lib/x86/Release
			* ../dist/lib/amd64
				* ../dist/lib/amd64/Debug
				* ../dist/lib/amd64/Release
		* ../dist/data
		* ../dist/doc

It calls all the **win-build_branch.pl** scripts of all the sub-projects defined in **win-init_env_command.pl**,
bin, include, lib, data, doc directories of all theses project will be copied in ./dist.

In the scripts directory, you can modify win-init_env_command.pl in order to change the libs paths of the project


2. CHOOSE project to be built
-----------------------------

You can set the Reservoir-cuda modules to be built by modifiying the **PBuildOrder** variable in **win-init_env_command.pl** :

        my @PbuildOrder  = ($Reservoir, $Other_project1, $Other_project2);

Here The first project to be built is Reservoir and after that Other_project1.

        my @PbuildOrder  = ($Reservoir, $Other_project2);

Now Other_project1 is excluded from the Reservoir-cuda project build.
The order is important, some projects depends from each others.


	
3. CLEAN RESERVOIR-CUDA
-----------------------
	
win-clean.pl will delete all the content of the dist repertory (except dist/data) and all the compiled files
in the swooz projects defined in @PbuildOrder.


4. GENERATE DOC
---------------
 
win-doc-generate.pl will call all the win-generate_doc.pl of each project, these scripts generate the Doxygen documentation using their respective Doxyfile.

5. CLEAN DOC
------------

win-doc-clean.pl will delete the doc directories of all the projects.



