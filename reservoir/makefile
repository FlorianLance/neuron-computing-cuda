# makefile of swooz-teleoperation project using Visual C++
# Guillaume Gibert / Florian Lance
# lyon sept 2013

!IF "$(CFG)" != "Release" &&  "$(CFG)" != "Debug"
!ERROR Invalid CFG "($CFG)
!ENDIF

!IF "$(ARCH)" != "x86" &&  "$(ARCH)" != "amd64"
!ERROR Invalid ARCH "($ARCH)
!ENDIF

CC=cl
LINK=link

!INCLUDE <./makefile-include>


############################################################################## OBJ LISTS

RESERVOIR_OBJ=\
    $(LIBDIR)/Reservoir.obj $(LIBDIR)/Model.obj $(LIBDIR)/inversions.obj $(LIBDIR)/multiplications.obj $(LIBDIR)/GridSearch.obj $(LIBDIR)/Generalization.obj $(LIBDIR)/main.obj\

RESERVOIR_OBJ_S=\
    $(LIBDIR)/Reservoir-s.obj $(LIBDIR)/Model-s.obj $(LIBDIR)/inversions.obj $(LIBDIR)/multiplications.obj $(LIBDIR)/GridSearch-s.obj $(LIBDIR)/Generalization-s.obj $(LIBDIR)/main-s.obj\

############################################################################## Makefile commands

!if "$(ARCH)" == "x86"
all: $(BINDIR)/reservoir-test.exe
!endif

!if "$(ARCH)" == "amd64"
all: $(BINDIR)/reservoir-test-x64.exe
!endif

############################################################################## exe files

WIN_CONFIG = $(SETARGV) $(BINMODE) $(WINLIBS)

# exec files to be generated
$(BINDIR)/reservoir-test.exe: $(RESERVOIR_OBJ) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir-test.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_OBJ) $(LIBS_RESERVOIR) $(WIN_CONFIG)

$(BINDIR)/reservoir-test-x64.exe: $(RESERVOIR_OBJ) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir-test-x64.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_OBJ) $(LIBS_RESERVOIR) $(WIN_CONFIG)

##################################################### sources files

$(LIBDIR)/Reservoir.obj: ./src/Reservoir.cpp
        $(CC) -c ./src/Reservoir.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Reservoir.obj"

$(LIBDIR)/Reservoir-s.obj: ./src/Reservoir.cpp
        $(CC) -c ./src/Reservoir.cpp $(CFLAGS_STA) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Reservoir-s.obj"


$(LIBDIR)/Model.obj: ./src/Model.cpp
        $(CC) -c ./src/Model.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Model.obj"

$(LIBDIR)/Model-s.obj: ./src/Model.cpp
        $(CC) -c ./src/Model.cpp $(CFLAGS_STA) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Model-s.obj"


$(LIBDIR)/GridSearch.obj: ./src/GridSearch.cpp
        $(CC) -c ./src/GridSearch.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/GridSearch.obj"

$(LIBDIR)/GridSearch-s.obj: ./src/GridSearch.cpp
        $(CC) -c ./src/GridSearch.cpp $(CFLAGS_STA) $(RESERVOIR_INC) -Fo"$(LIBDIR)/GridSearch-s.obj"


$(LIBDIR)/Generalization.obj: ./src/Generalization.cpp
        $(CC) -c ./src/Generalization.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Generalization.obj"

$(LIBDIR)/Generalization-s.obj: ./src/Generalization.cpp
        $(CC) -c ./src/Generalization.cpp $(CFLAGS_STA) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Generalization-s.obj"


$(LIBDIR)/main.obj: ./src/main.cpp
        $(CC) -c ./src/main.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/main.obj"

$(LIBDIR)/main-s.obj: ./src/main.cpp
        $(CC) -c ./src/main.cpp $(CFLAGS_STA) $(RESERVOIR_INC) -Fo"$(LIBDIR)/main-s.obj"


