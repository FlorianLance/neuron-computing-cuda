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
    $(LIBDIR)/Generalization.obj $(LIBDIR)/Reservoir.obj $(LIBDIR)/Model.obj $(LIBDIR)/inversions.obj $(LIBDIR)/multiplications.obj $(LIBDIR)/GridSearch.obj\

RESERVOIR_COMMAND_OBJ=\
    $(RESERVOIR_OBJ) $(LIBDIR)/main.obj\

RESERVOIR_YARP_OBJ=\
    $(RESERVOIR_OBJ) $(LIBDIR)/YarpInterface.obj\

RESERVOIR_QT_INTERFACE_OBJ=\
    $(RESERVOIR_OBJ) $(LIBDIR)/qcustomplot.obj $(LIBDIR)/Interface.obj $(LIBDIR)/InterfaceWorker.obj\

RESERVOIR_YARP_TEST_OBJ=\
    $(RESERVOIR_OBJ) $(LIBDIR)/TestYarp.obj\

############################################################################## MOC LIST

RESERVOIR_MOC=\
    $(MOCDIR)/moc_Interface.cpp $(MOCDIR)/moc_InterfaceWorker.cpp $(MOCDIR)/moc_Reservoir.cpp $(MOCDIR)/moc_Model.cpp $(MOCDIR)/moc_GridSearch.cpp\
    $(MOCDIR)/moc_qcustomplot.cpp $(MOCDIR)/moc_YarpInterface.cpp\

############################################################################## Makefile commands

!if "$(ARCH)" == "x86"
all: $(QTGENW)/UI_Interface.h $(RESERVOIR_MOC) $(BINDIR)/reservoir-interface.exe $(BINDIR)/reservoir.exe $(BINDIR)/reservoir-yarp.exe
!endif

!if "$(ARCH)" == "amd64"
all: $(QTGENW)/UI_Interface.h $(RESERVOIR_MOC) $(BINDIR)/reservoir-interface-x64.exe $(BINDIR)/reservoir-x64.exe $(BINDIR)/reservoir-yarp-x64.exe $(BINDIR)/test-yarp-x64.exe
!endif

############################################################################## exe files

WIN_CONFIG = $(SETARGV) $(BINMODE) $(WINLIBS)


$(BINDIR)/reservoir-interface.exe: $(RESERVOIR_QT_INTERFACE_OBJ) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir-interface.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_QT_INTERFACE_OBJ) $(LIBS_RESERVOIR) $(WIN_CONFIG)

$(BINDIR)/reservoir-interface-x64.exe: $(RESERVOIR_QT_INTERFACE_OBJ) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir-interface-x64.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_QT_INTERFACE_OBJ) $(LIBS_RESERVOIR) $(WIN_CONFIG)

$(BINDIR)/reservoir.exe: $(RESERVOIR_COMMAND_OBJ) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_COMMAND_OBJ) $(LIBS_RESERVOIR) $(WIN_CONFIG)

$(BINDIR)/reservoir-x64.exe: $(RESERVOIR_COMMAND_OBJ) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir-x64.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_COMMAND_OBJ) $(LIBS_RESERVOIR) $(WIN_CONFIG)

$(BINDIR)/reservoir-yarp.exe: $(RESERVOIR_YARP_OBJ) $(LIBS_RESERVOIR_YARP)
        $(LINK) /OUT:$(BINDIR)/reservoir-yarp.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_YARP_OBJ) $(LIBS_RESERVOIR_YARP) $(WIN_CONFIG)

$(BINDIR)/reservoir-yarp-x64.exe: $(RESERVOIR_YARP_OBJ) $(LIBS_RESERVOIR_YARP)
        $(LINK) /OUT:$(BINDIR)/reservoir-yarp-x64.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_YARP_OBJ) $(LIBS_RESERVOIR_YARP) $(WIN_CONFIG)

$(BINDIR)/test-yarp-x64.exe: $(RESERVOIR_YARP_TEST_OBJ) $(LIBS_RESERVOIR_YARP)
        $(LINK) /OUT:$(BINDIR)/test-yarp-x64.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_YARP_TEST_OBJ) $(LIBS_RESERVOIR_YARP) $(WIN_CONFIG)

##################################################### sources files

$(LIBDIR)/Reservoir.obj: ./src/Reservoir.cpp
        $(CC) -c ./src/Reservoir.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Reservoir.obj"

$(LIBDIR)/Model.obj: ./src/Model.cpp
        $(CC) -c ./src/Model.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Model.obj"

$(LIBDIR)/GridSearch.obj: ./src/GridSearch.cpp
        $(CC) -c ./src/GridSearch.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/GridSearch.obj"

$(LIBDIR)/Generalization.obj: ./src/Generalization.cpp
        $(CC) -c ./src/Generalization.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Generalization.obj"

$(LIBDIR)/main.obj: ./src/main.cpp
        $(CC) -c ./src/main.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/main.obj"

$(LIBDIR)/Interface.obj: ./src/Interface.cpp
        $(CC) -c ./src/Interface.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Interface.obj"

$(LIBDIR)/InterfaceWorker.obj: ./src/InterfaceWorker.cpp
        $(CC) -c ./src/InterfaceWorker.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/InterfaceWorker.obj"

$(LIBDIR)/YarpInterface.obj: ./src/YarpInterface.cpp
        $(CC) -c ./src/YarpInterface.cpp $(CFLAGS_DYN) $(RESERVOIR_YARP) -Fo"$(LIBDIR)/YarpInterface.obj"

$(LIBDIR)/qcustomplot.obj: ./src/qcustomplot.cpp
        $(CC) -c ./src/qcustomplot.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/qcustomplot.obj"

$(LIBDIR)/TestYarp.obj: ./src/TestYarp.cpp
        $(CC) -c ./src/TestYarp.cpp $(CFLAGS_DYN) $(RESERVOIR_YARP) -Fo"$(LIBDIR)/TestYarp.obj"

############################################################################## Qt ui files

$(QTGENW)/UI_Interface.h: $(FORMDIR)/interfaceReservoir.ui
        uic $(FORMDIR)/interfaceReservoir.ui -o $@

############################################################################## Qt moc files

$(MOCDIR)/moc_Interface.cpp: ./include/Interface.h
         moc  ./include/Interface.h -o $@

$(MOCDIR)/moc_InterfaceWorker.cpp: ./include/InterfaceWorker.h
         moc  ./include/InterfaceWorker.h -o $@

$(MOCDIR)/moc_Reservoir.cpp: ./include/Reservoir.h
         moc  ./include/Reservoir.h -o $@

$(MOCDIR)/moc_Model.cpp: ./include/Model.h
         moc  ./include/Model.h -o $@

$(MOCDIR)/moc_GridSearch.cpp: ./include/GridSearch.h
         moc  ./include/GridSearch.h -o $@

$(MOCDIR)/moc_YarpInterface.cpp: ./include/YarpInterface.h
         moc  ./include/YarpInterface.h -o $@

$(MOCDIR)/moc_qcustomplot.cpp: ./include/qcustomplot.h
         moc  ./include/qcustomplot.h -o $@
