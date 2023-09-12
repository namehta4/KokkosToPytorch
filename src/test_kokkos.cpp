#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/embed.h>
#include<Kokkos_Core.hpp>
#include<iostream>
#include<random>
#include<cmath>
#include<cstdio>
#include<stdio.h>

#if defined(KOKKOS_ENABLE_CUDA)
#include<cuda.h>
#include<cuda_runtime.h>
#endif

#if defined(KOKKOS_ENABLE_HIP)
#include<hip/hip_runtime.h>
#endif

using namespace std;

typedef Kokkos::View<double **, Kokkos::LayoutRight> View2D;
typedef View2D::HostMirror h_View2D;

void kokkos_loop(pybind11::array_t<double> pyfj, int N, int D)
{
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace, \
  Kokkos::MemoryTraits<Kokkos::Unmanaged>> foo1;
  View2D foo1_t("foo1_t",N,D);
  h_View2D h_foo1;
  h_foo1 = create_mirror_view(foo1_t);
  
  pybind11::buffer_info buf1 = pyfj.request();
  foo1 = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace, \
  Kokkos::MemoryTraits<Kokkos::Unmanaged>> (static_cast<double *>(buf1.ptr),N,D);
  Kokkos::fence();

  Kokkos::parallel_for(N*D, KOKKOS_LAMBDA(const int iter)
  {
    int i = iter / D;
    int j = iter % D;
    foo1(i,j) += 1;
  });
  deep_copy(h_foo1,foo1);
  Kokkos::fence();
  fprintf(stdout,"C++ increment: Value of foo1(10,3) is %f \n",h_foo1(N-1,D-1));

  // Calling loop implemented in Python
  pybind11::module sys = pybind11::module::import("sys");
  sys.attr("path").attr("insert")(1, CUSTOM_SYS_PATH);
  pybind11::module py_simple = pybind11::module::import("py_simple");
  pybind11::object ob1 = py_simple.attr("add_NN")(pybind11::array_t\
      	    <double, pybind11::array::c_style | pybind11::array::forcecast>\
      	    (N*D,foo1.data(),pybind11::str{}),N,D);
  deep_copy(h_foo1,foo1);
  Kokkos::fence();
  fprintf(stdout,"PY_SIMPLE increment: Value of foo1(10,3) is %f \n",h_foo1(N-1,D-1));
}

int kokkos_begin()
{
  Kokkos::initialize();
  return 0;
}

int kokkos_end()
{
  Kokkos::finalize();
  return 0;
}

PYBIND11_MODULE(test_kokkos, m)
{
  m.doc() = "Test for kokkos array back and forth";
  m.def("kokkos_loop", &kokkos_loop, "Python 2 kokkos 2 python");
  m.def("kokkos_begin", &kokkos_begin, "Initialize kokkos from python");
  m.def("kokkos_end", &kokkos_end, "Finalize kokkos from python");  
}

