#define NULL 0

#include <iostream>
#include <complex>
#include <assert.h>
#include <algorithm>    // std::find
#include "statecreps.h"
//#include <pthread.h>

//using namespace std::complex_literals;

//#define DEBUG(x) x
#define DEBUG(x) 

namespace CReps_densitymx {

  /****************************************************************************\
  |* StateCRep                                                              *|
  \****************************************************************************/
  StateCRep::StateCRep(INT dim) {
    //DEBUG std::cout << "densitymx.StateCRep initialized w/dim = " << dim << std::endl;
    _dataptr = new double[dim];
    for(INT i=0; i<dim; i++) _dataptr[i] = 0;
    _dim = dim;
    _ownmem = true;
  }
  
  StateCRep::StateCRep(double* data, INT dim, bool copy) {
    //DEBUG std::cout << "StateCRep initialized w/dim = " << dim << std::endl;
    if(copy) {
      _dataptr = new double[dim];
      for(INT i=0; i<dim; i++) {
	_dataptr[i] = data[i];
      }
    } else {
      _dataptr = data;
    }
    _dim = dim;
    _ownmem = copy;
  }

  StateCRep::~StateCRep() {
    if(_ownmem && _dataptr != NULL)
      delete [] _dataptr;
  }

  void StateCRep::print(const char* label) {
    std::cout << label << " = [";
    for(INT i=0; i<_dim; i++) std::cout << _dataptr[i] << " ";
    std::cout << "]" << std::endl;
  }

  void StateCRep::copy_from(StateCRep* st) {
    assert(_dim == st->_dim);
    for(INT i=0; i<_dim; i++)
      _dataptr[i] = st->_dataptr[i];
  }
}
