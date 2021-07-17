#include <pybind11/pybind11.h>
#include <omp.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
};

void fma(int n, double * out, double const * weights, double const * array) {
    # pragma omp simd
    for (int i = 0; i < n; ++i) {
        out[i] += weights[i] * array[i];
    }
};

void fma_scalar_weight(int n, double * out, double const weight, double const * array) {
    # pragma omp simd
    for (int i = 0; i < n; ++i) {
        out[i] += weight * array[i];
    }
}

void fma_vector_weights_arrays(int n_out, int n_weights, double * out, double const * weights, double ** arrays) {
    # pragma omp parallel for simd
    for (int i = 0; i < n_weights; ++i) {
        double const weight = weights[i];
        double * const array = arrays[i];
        for (int j = 0; j < n_out; ++j) {
            out[j] += weight * array[j];
        }
    }
}

namespace py = pybind11;

void fma_warp(py::buffer out, py::buffer weights, py::buffer array) {
    py::buffer_info info_out = out.request();
    py::buffer_info info_weights = weights.request();
    py::buffer_info info_array = array.request();

    double * out_raw = reinterpret_cast <double *> (info_out.ptr);
    double * weights_raw = reinterpret_cast <double *> (info_weights.ptr);
    double * array_raw = reinterpret_cast <double *> (info_array.ptr);

    if (info_out.size != info_weights.size || info_out.size != info_array.size) {
        throw std::runtime_error("Buffers are different sizes");
    }

    fma(info_array.size, out_raw, weights_raw, array_raw);
};

void fma_scalar_weight_warp(py::buffer out, double weight, py::buffer array) {
    py::buffer_info info_out = out.request();
    py::buffer_info info_array = array.request();

    double * out_raw = reinterpret_cast <double *> (info_out.ptr);
    double * array_raw = reinterpret_cast <double *> (info_array.ptr);

    if (info_out.size != info_array.size) {
        throw std::runtime_error("Buffers are different sizes");
    }

    fma_scalar_weight(info_array.size, out_raw, weight, array_raw);
};

void fma_vector_weights_arrays_wrap(py::buffer out, py::buffer weights, py::args a) {
    py::buffer_info info_out = out.request();
    py::buffer_info info_weights = weights.request();

    if ((unsigned) info_weights.size != a.size()) {
        throw std::runtime_error("Weights & args are different sizes");
    }

    double * out_raw = reinterpret_cast <double *> (info_out.ptr);
    double * weights_raw = reinterpret_cast <double *> (info_weights.ptr);

    double** arrays = new double*[a.size()];

    for (size_t i = 0; i < a.size(); ++i) {
        // Use raw Python API here to avoid an extra, intermediate incref on the tuple item:
        py::handle array = PyTuple_GET_ITEM(a.ptr(), static_cast<py::ssize_t>(i));
        py::buffer array_buffer = array.cast<py::buffer>();
        py::buffer_info info_array = array_buffer.request();

        if (info_array.size != info_out.size) {
            throw std::runtime_error("Buffers are different sizes");
        }

        arrays[i] = reinterpret_cast <double *> (info_array.ptr);
    }
    fma_vector_weights_arrays(info_out.size, info_weights.size, out_raw, weights_raw, arrays);
    delete [] arrays;
}

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("fma", &fma_warp, R"pbdoc(
        Simple fused multiply–add, compiled to avoid Python memory implications.

        :param out: output array, summed in-place.
        :param weights: weights of the input array.
        :param array: input array.
    )pbdoc");

    m.def("fma_scalar_weight", &fma_scalar_weight_warp, R"pbdoc(
        Simple fused multiply–add, compiled to avoid Python memory implications.

        :param out: output array, summed in-place.
        :param weight: weight of the input array.
        :param array: input array.
    )pbdoc");

    m.def("fma_vector_weights", [](py::buffer out, py::buffer weights, py::args a) {
        py::buffer_info info_weights = weights.request();
        double * weights_raw = reinterpret_cast <double *> (info_weights.ptr);

        if ((unsigned) info_weights.size != a.size()) {
            throw std::runtime_error("Buffers are different sizes");
        }

        for (size_t i = 0; i < a.size(); ++i) {
            // Use raw Python API here to avoid an extra, intermediate incref on the tuple item:
            py::handle array = PyTuple_GET_ITEM(a.ptr(), static_cast<py::ssize_t>(i));
            py::buffer array_buffer = array.cast<py::buffer>();
            fma_scalar_weight_warp(out, weights_raw[i], array_buffer);
        }
    }, R"pbdoc(
        Simple fused multiply–add, compiled to avoid Python memory implications.

        :param out: output array, summed in-place.
        :param weights: weights of the input array.
        :param arrays: list of input arrays.
    )pbdoc");

    m.def("fma_vector_weights_arrays", &fma_vector_weights_arrays_wrap, R"pbdoc(
        Simple fused multiply–add, compiled to avoid Python memory implications.

        :param out: output array, summed in-place.
        :param weights: weights of the input array.
        :param arrays: list of input arrays.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
