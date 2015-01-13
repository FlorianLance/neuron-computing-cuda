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
    $(LIBDIR)/Generalization.obj $(LIBDIR)/ReservoirQtObject.obj $(LIBDIR)/ModelQtObject.obj $(LIBDIR)/inversions.obj $(LIBDIR)/multiplications.obj $(LIBDIR)/GridSearchQtObject.obj $(LIBDIR)/main.obj\

RESERVOIR_RFMODULE=\
    $(RESERVOIR_OBJ)

RESERVOIR_COMMAND=\
    $(RESERVOIR_OBJ)

RESERVOIR_QT_INTERFACE_OBJ=\
    $(LIBDIR)/ReservoirQtObject.obj $(LIBDIR)/ModelQtObject.obj $(LIBDIR)/inversions.obj $(LIBDIR)/multiplications.obj\
    $(LIBDIR)/qcustomplot.obj $(LIBDIR)/GridSearchQtObject.obj $(LIBDIR)/Interface.obj $(LIBDIR)/InterfaceWorker.obj\

############################################################################## MOC LIST

RESERVOIR_MOC=\
    $(MOCDIR)/moc_Interface.cpp $(MOCDIR)/moc_InterfaceWorker.cpp $(MOCDIR)/moc_ReservoirQtObject.cpp $(MOCDIR)/moc_ModelQtObject.cpp $(MOCDIR)/moc_GridSearchQtObject.cpp\
    $(MOCDIR)/moc_qcustomplot.cpp\

############################################################################## Makefile commands

!if "$(ARCH)" == "x86"
all: $(QTGENW)/UI_Interface.h $(RESERVOIR_MOC) $(BINDIR)/reservoir-interface.exe $(BINDIR)/reservoir.exe
!endif

!if "$(ARCH)" == "amd64"
all: $(QTGENW)/UI_Interface.h $(RESERVOIR_MOC) $(BINDIR)/reservoir-interface-x64.exe $(BINDIR)/reservoir-x64.exe
!endif

############################################################################## exe files

WIN_CONFIG = $(SETARGV) $(BINMODE) $(WINLIBS)

# exec files to be generated
$(BINDIR)/reservoir-interface.exe: $(RESERVOIR_QT_INTERFACE_OBJ) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir-interface.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_QT_INTERFACE_OBJ) $(LIBS_RESERVOIR) $(WIN_CONFIG)

$(BINDIR)/reservoir-interface-x64.exe: $(RESERVOIR_QT_INTERFACE_OBJ) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir-interface-x64.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_QT_INTERFACE_OBJ) $(LIBS_RESERVOIR) $(WIN_CONFIG)

# exec files to be generated
$(BINDIR)/reservoir.exe: $(RESERVOIR_COMMAND) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_COMMAND) $(LIBS_RESERVOIR) $(WIN_CONFIG)

$(BINDIR)/reservoir-x64.exe: $(RESERVOIR_COMMAND) $(LIBS_RESERVOIR)
        $(LINK) /OUT:$(BINDIR)/reservoir-x64.exe $(LFLAGS_RESERVOIR) $(RESERVOIR_COMMAND) $(LIBS_RESERVOIR) $(WIN_CONFIG)

##################################################### sources files

$(LIBDIR)/ReservoirQtObject.obj: ./src/ReservoirQtObject.cpp
        $(CC) -c ./src/ReservoirQtObject.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/ReservoirQtObject.obj"

$(LIBDIR)/ModelQtObject.obj: ./src/ModelQtObject.cpp
        $(CC) -c ./src/ModelQtObject.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/ModelQtObject.obj"

$(LIBDIR)/GridSearchQtObject.obj: ./src/GridSearchQtObject.cpp
        $(CC) -c ./src/GridSearchQtObject.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/GridSearchQtObject.obj"

$(LIBDIR)/Generalization.obj: ./src/Generalization.cpp
        $(CC) -c ./src/Generalization.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Generalization.obj"

$(LIBDIR)/main.obj: ./src/main.cpp
        $(CC) -c ./src/main.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/main.obj"

$(LIBDIR)/Interface.obj: ./src/Interface.cpp
        $(CC) -c ./src/Interface.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/Interface.obj"

$(LIBDIR)/InterfaceWorker.obj: ./src/InterfaceWorker.cpp
        $(CC) -c ./src/InterfaceWorker.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/InterfaceWorker.obj"

$(LIBDIR)/qcustomplot.obj: ./src/qcustomplot.cpp
        $(CC) -c ./src/qcustomplot.cpp $(CFLAGS_DYN) $(RESERVOIR_INC) -Fo"$(LIBDIR)/qcustomplot.obj"


############################################################################## Qt ui files

$(QTGENW)/UI_Interface.h: $(FORMDIR)/interfaceReservoir.ui
        uic $(FORMDIR)/interfaceReservoir.ui -o $@

############################################################################## Qt moc files

$(MOCDIR)/moc_Interface.cpp: ./include/Interface.h
         moc  ./include/Interface.h -o $@

$(MOCDIR)/moc_InterfaceWorker.cpp: ./include/InterfaceWorker.h
         moc  ./include/InterfaceWorker.h -o $@

$(MOCDIR)/moc_ReservoirQtObject.cpp: ./include/ReservoirQtObject.h
         moc  ./include/ReservoirQtObject.h -o $@

$(MOCDIR)/moc_ModelQtObject.cpp: ./include/ModelQtObject.h
         moc  ./include/ModelQtObject.h -o $@

$(MOCDIR)/moc_GridSearchQtObject.cpp: ./include/GridSearchQtObject.h
         moc  ./include/GridSearchQtObject.h -o $@

$(MOCDIR)/moc_qcustomplot.cpp: ./include/qcustomplot.h
         moc  ./include/qcustomplot.h -o $@
