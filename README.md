# era5_to_int
A simple utility for converting ERA5 netCDF files to the WPS intermediate format.

This has been packaged up and slightly modified from the original [NCAR era5_to_int scripts](https://github.com/NCAR/era5_to_int). Huge thanks to NCAR and to Michael Duda and Anthony Islas for writing the scripts!

## Installation
You can install the command line utility in a variety of ways. One of the easiest and most common is to use pipx:
```
pipx install era5_to_int
```

## Overview

The `era5_to_int` utility converts ERA5 pressure-level netCDF files
to the WRF Pre-processing System (WPS) intermediate file format, permitting the
use of these data with either the Weather Research and Forecasting (WRF) model
or the Model for Prediction Across Scales - Atmosphere (MPAS-A).

The required command-line arguments are the path to the local files and the start and end datetimes of ERA5 files to convert. For example:
```
era5_to_int /root/path/to/era5/ 2024-05-01_00 2024-05-02_00
```

or if you're using a different time format, then you'll need to put quotes around the datetimes:
```
era5_to_int /root/path/to/era5/ "2024-05-01 00" "2024-05-02 00"
```

The directory structure must be the same as [NCAR's ERA5 AWS S3 bucket](https://nsf-ncar-era5.s3.amazonaws.com/index.html). This is the recommended place to download the ERA5 data since it's free and fast.

Conversion of a range of datetimes is possible through the use of additional
command-line arguments as described in the Usage section.

The following surface fields from the d633 datasets are handled by the script:

| Field   | Dataset | Horiz. grid | Num. levels |
|---------|---------|-------------|-------------|
| SOILGEO | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| SP      | [d633006](https://rda.ucar.edu/datasets/d633006/) | 0.25-deg Lat-Lon | 1 |
| MSL     | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| VAR_2T  | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| VAR_2D  | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| VAR_10U | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| VAR_10V | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| RSN     | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| SD      | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| LSM     | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| SSTK    | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| SKT     | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| SWVL1   | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| SWVL2   | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| SWVL3   | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| SWVL4   | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| STL1    | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| STL2    | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| STL3    | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| STL4    | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |
| CI      | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 1 |

When pressure-level (isobaric) ERA5 files are being processed, the following
atmospheric fields are handled:

| Field   | Dataset | Horiz. grid | Num. levels |
|---------|---------|-------------|-------------|
| Z       | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 37 |
| Q       | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 37 |
| T       | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 37 |
| U       | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 37 |
| V       | [d633000](https://rda.ucar.edu/datasets/d633000/) | 0.25-deg Lat-Lon | 37 |

As fields are converted from netCDF to intermediate format, their names are also
converted to match WPS and MPAS-A expectations.

## Downloading and spatial clipping the ERA5 data
This command line utility isn't very helpful if you don't have the ERA5 data. There is a docker image made specifically to download the ERA5 data from the AWS S3 bucket maintained by NCAR. Then the container will clip the global files to a user-defined region to substantially shrink the files. It will save the clipped files to a directory or optionally upload it to another S3 system defined by the user.

You can find the github repo here: 
https://github.com/envlib/era5-download

It will download all of the files necessary to run the era5_to_int utility.

## Usage
The required command-line arguments are the path to the local files and the start and end datetimes of ERA5 files to convert.
For example:
```
era5_to_int /root/path/to/era5/ 2024-05-01_00 2024-05-02_00
```

Upon successful completion, an intermediate file with the prefix `ERA5` is
created in the current working directory; for example, `ERA5:2024-05-01_00`.

By default, fields are converted at a six-hourly interval, and a different
interval may be specified with the option -h. For example, to
convert data at a three-hourly interval for the month of May 2024:
```
era5_to_int -h 3 /root/path/to/era5/ 2024-05-01_00 2024-05-02_00
```

Also by default, all variables required to run WRF are processed. If only a selection of variables is required, then use the --variables option:
```
era5_to_int --variables LANDSEA,SEAICE,SKINTEMP /root/path/to/era5/ 2024-05-01_00 2024-05-02_00
```

Usage help is provided by running the `era5_to_int` with the `--help`
argument.

