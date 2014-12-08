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
    $(LIBDIR)/Reservoir.obj $(LIBDIR)/Model.obj $(LIBDIR)/inversions.obj $(LIBDIR)/multiplications.obj $(LIBDIR)/GridSearch.obj $(LIBDIR)/Generalization.obj $(LIBDIR)/main.obj $(LIBDIR)/Interface.obj\
    $(LIBDIR)/ReservoirQtObject.obj $(LIBDIR)/ModelQtObject.obj $(LIBDIR)/GridSearchQtObject.obj\

############################################################################## MOC LIST

RESERVOIR_MOC=\
    $(MOCDIR)/moc_Interface.cpp $(MOCDIR)/moc_ReservoirQtObject.cpp $(MOCDIR)/moc_ModelQtObject.cpp $(MOCDIR)/moc_GridSearchQtObject.cpp\
    $(MOCDIR)/moc_DisplayImageWidget.cpp\

############################################################################## Makefile commands

!if "$(ARCH)" == "x86"
all: $(QTGENW)/UI_Interface.h $(RESERVOIR_MOC) $(BINDIR)/reservoir-test.exe
!endif

!if "$(ARCH)" == "amd64"
all: $(QTGENW)/UI_Interface.h $(RESERVOIR_MOC) $(BINDIR)/reservoir-test-x64.exe
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

$(LIBDIR)/ReservoirQtObject.obj: ./src/ReservoirQtObject.cpp
        $(CC) -c ./src/ReservoirQtObject.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/ReservoirQtObject.obj"

$(LIBDIR)/Model.obj: ./src/Model.cpp
        $(CC) -c ./src/Model.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Model.obj"

$(LIBDIR)/ModelQtObject.obj: ./src/ModelQtObject.cpp
        $(CC) -c ./src/ModelQtObject.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/ModelQtObject.obj"

$(LIBDIR)/GridSearch.obj: ./src/GridSearch.cpp
        $(CC) -c ./src/GridSearch.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/GridSearch.obj"

$(LIBDIR)/GridSearchQtObject.obj: ./src/GridSearchQtObject.cpp
        $(CC) -c ./src/GridSearchQtObject.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/GridSearchQtObject.obj"

$(LIBDIR)/Generalization.obj: ./src/Generalization.cpp
        $(CC) -c ./src/Generalization.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Generalization.obj"

$(LIBDIR)/main.obj: ./src/main.cpp
        $(CC) -c ./src/main.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/main.obj"

$(LIBDIR)/Interface.obj: ./src/Interface.cpp
        $(CC) -c ./src/Interface.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Interface.obj"

$(LIBDIR)/DisplayImageWidget.obj: ./src/DisplayImageWidget.cpp
        $(CC) -c ./src/DisplayImageWidget.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/DisplayImageWidget.obj"

############################################################################## Qt ui files

$(QTGENW)/UI_Interface.h: $(FORMDIR)/interfaceReservoir.ui
        uic $(FORMDIR)/interfaceReservoir.ui -o $@

############################################################################## Qt moc files

$(MOCDIR)/moc_Interface.cpp: ./include/Interface.h
         moc  ./include/Interface.h -o $@

$(MOCDIR)/moc_ReservoirQtObject.cpp: ./include/ReservoirQtObject.h
         moc  ./include/ReservoirQtObject.h -o $@

$(MOCDIR)/moc_ModelQtObject.cpp: ./include/ModelQtObject.h
         moc  ./include/ModelQtObject.h -o $@

$(MOCDIR)/moc_GridSearchQtObject.cpp: ./include/GridSearchQtObject.h
         moc  ./include/GridSearchQtObject.h -o $@

$(MOCDIR)/moc_DisplayImageWidget.cpp: ./include/DisplayImageWidget.h
         moc  ./include/DisplayImageWidget.h -o $@
