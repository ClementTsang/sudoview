#include "sudosolver.h"
#include <iostream>
#include <Python.h>
#include <stdlib.h>

using namespace std;

int main() {
    string image;
    cout << "Enter path to image: "
    cin >> image;
    Py_SetProgramName(argv[0]);
    Py_Initalize();
    pName = PyString_FromString((char*)"pyocr");
    pModule = PyImport_Import(pName);
    pFunc = PyDict_GetItemString(pDict, (char*)"getArr");
    
    param = Py_BuildValue("(z)", (char*)image)
    result = PyObject_CallObject(pFunc, param);

    Py_Finalize();
}