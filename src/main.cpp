#include<hip/hip_runtime.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/embed.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<iostream>
#include<random>
#include<cmath>
#include<cstdio>

//Training batch size
const int64_t N = 64;
//Tensor input dimension
const int64_t D_in = 1000;

using namespace std;
namespace py = pybind11;

typedef Kokkos::View<double *, Kokkos::LayoutRight> View1D;
typedef Kokkos::View<double **, Kokkos::LayoutRight> View2D;
typedef View2D::HostMirror h_View2D;


int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    View2D foo1("Foo1", N,D_in);
    h_View2D h_foo1;
    h_foo1 = create_mirror_view(foo1);

    Kokkos::parallel_for(N*D_in, KOKKOS_LAMBDA(const int iter)
    {
      int i = iter / D_in;
      int j = iter % D_in;
      foo1(i,j) = i*j;
    });
    deep_copy(h_foo1,foo1);
    printf("Before: Value of foo1(60,60) is %f \n",h_foo1(60,60));


// Calling simple NN implemented in Python
    py::scoped_interpreter guard{};
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, CUSTOM_SYS_PATH);
    py::module py_simple = py::module::import("py_simple");
    py::object ob1 = py_simple.attr("add_NN")(py::array_t<double, py::array::c_style | py::array::forcecast>(N*D_in,foo1.data(),py::str{}),N,D_in);
    py::gil_scoped_release no_gil;
    
    deep_copy(h_foo1,foo1);
    printf("After: Value of foo1(60,60) is %f \n",h_foo1(60,60));
  }
  Kokkos::finalize();
  return 0;
}


