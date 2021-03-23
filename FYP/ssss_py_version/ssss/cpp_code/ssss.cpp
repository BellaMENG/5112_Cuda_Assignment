#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>
#include <iostream>
#include <string>
#include "split_interpolation.hpp"

using namespace boost::python;
using namespace std;

char const* foo(int x, int y) {return "foo";}

boost::python::list encrypt_str(std::string secret, int intercept, int degree) {
    string shares[intercept];
    encrypt_string(shares, secret, intercept, degree);
    boost::python::list list_shares;
    for (int i = 0; i < intercept; ++i) {
        list_shares.append(shares[i]);
    }
    return list_shares;
}


boost::python::list encrypt_str_cheating(std::string secret, int intercept, int degree) {
    string shares_f[intercept];
    string shares_g[intercept];
    encrypt_string_cheating(shares_f, shares_g, secret, intercept, degree);
    boost::python::list list_shares;
    for (int i = 0; i < intercept; ++i) {
        list_shares.append(shares_f[i]);
    }
    for (int i = 0; i < intercept; ++i) {
        list_shares.append(shares_g[i]);
    }
    return list_shares;
}


boost::python::list encrypt_int_px(boost::python::list pixels_list, int intercept, int degree) {
    vector<int> pixels;
    for (int i = 0; i < boost::python::len(pixels_list); ++i) {
        pixels.push_back(boost::python::extract<int>(pixels_list[i]));
    }
    vector< vector<int> > shares;
    encrypt_pic(shares, pixels, intercept, degree);
    boost::python::list list_shares;
    for (int i = 0; i < intercept; ++i) {
        boost::python::list list_share;
        for (int j = 0; j < pixels.size(); ++j) {
            list_share.append(shares[i][j]);
        }
        list_shares.append(list_share);
    }
    return list_shares;
}


string decrypt_str(boost::python::list shares, int degree) {
    string shares_arr[degree+1];
    for (int i = 0; i < degree+1; ++i)
        shares_arr[i] = boost::python::extract<std::string>(shares[i]);
    return reconstruct_secret(shares_arr, degree);
}


string decrypt_str_cheating(boost::python::list shares_f, boost::python::list shares_g, int degree) {
    string shares_arr_f[degree+1];
    string shares_arr_g[degree+1];
    for (int i = 0; i < degree+1; ++i) {
        shares_arr_f[i] = boost::python::extract<std::string>(shares_f[i]);
    }
    for (int i = 0; i < degree+1; ++i) {
        shares_arr_g[i] = boost::python::extract<std::string>(shares_g[i]);
    }
    return reconstruct_secret_cheating(shares_arr_f, shares_arr_g, degree);
}


boost::python::list decrypt_px(boost::python::list x_list, boost::python::list y_list, int degree) {
    vector<int> x_values;
    vector< vector<int> > y_values;
    for (int i = 0; i < degree+1; ++i) {
        x_values.push_back(boost::python::extract<int>(x_list[i]));
    }
    for (int i = 0; i < degree+1; ++i) {
        vector<int> y_value;
        for (int j = 0; j < boost::python::len(y_list[0]); ++j) {
            y_value.push_back(boost::python::extract<int>(y_list[i][j]));
        }
        y_values.push_back(y_value);
    }
    // call the function
    vector<int> secret;
    reconstruct_vect(secret, x_values, y_values, degree);
    boost::python::list secret_list;
    for (int i = 0; i < secret.size(); ++i) {
        secret_list.append(secret[i]);
    }
    return secret_list;
}


BOOST_PYTHON_MODULE(ssss)
{
    boost::python::def("foo", foo, boost::python::args("x", "y"), "foo's docstring");
    boost::python::def("encrypt_str", encrypt_str, boost::python::args("secret", "intercept", "degree"), "encrypt_function");
    boost::python::def("decrypt_str", decrypt_str, boost::python::args("shares", "degree"), "decrypt_function");
    boost::python::def("encrypt_str_cheating", encrypt_str_cheating, boost::python::args("secret", "intercept", "degree"), "encrypt_function with cheating detection");
    boost::python::def("decrypt_str_cheating", decrypt_str_cheating, boost::python::args("shares_f", "shares_f", "degree"), "decrypt_function with cd");
    boost::python::def("encrypt_int_px", encrypt_int_px, boost::python::args("pixels_list, intercept, degree"), "encrypt pixels");
    boost::python::def("decrypt_px", decrypt_px, boost::python::args("x_list", "y_list", "degree"), "decrypt image");
}