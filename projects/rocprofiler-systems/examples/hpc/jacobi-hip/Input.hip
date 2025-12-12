/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include <string.h>
#include "Jacobi.hpp"

/**
 * @file Input.c
 * @brief This contains the command-line argument parser and support functions
 */

// ====================================
// Command-line arguments parsing block
// ====================================

// Print the usage information for this application
void PrintUsage(const char * appName)
{
	printf("Usage: %s -g Grid.X [Grid.Y] [-m Mesh.X [Mesh.Y]] [-h | --help]\n", appName);
	printf(" -g Grid.x [Grid.y]: set the topology size (if \"Grid.y\" is missing, the topology will default to (Grid.x, 1); Grid.x and Grid.y must be positive integers)\n");
	printf(" -m Mesh.x [Mesh.y]: set the domain size per node (if \"Mesh.y\" is missing, the domain size will default to (Mesh.x, Mesh.x); Mesh.x and Mesh.y must be positive integers)\n");
	printf(" -h | --help: print help information\n");
}

// Find (and if found, erase) an argument in the command line
int FindAndClearArgument(const char * argName, int argc, char ** argv) {
	for(int i = 1; i < argc; ++i)	{
		if (strcmp(argv[i], argName) == 0) {
			strcpy(argv[i], "");
			return i;
		}
	}

	return -1;
}

// Extract a number given as a command-line argument
int ExtractNumber(int argIdx, int argc, char ** argv) {
	int result = 0;

	if (argIdx < argc) {
		result = atoi(argv[argIdx]);
		if (result > 0)	{
			strcpy(argv[argIdx], "");
		}
	}

	return result;
}

/**
 * @brief Parses the application's command-line arguments
 *
 * @param[in] argc          The number of input arguments
 * @param[in] argv        	The input arguments
 * @param[in] MPI_Comm      The MPI communicator
 * @param[out] grid         The MPI topology struct
 * @param[out] mesh         The mesh topology struct
 */
void ParseCommandLineArguments(int argc, char ** argv,
															MPI_Comm comm, grid_t& grid, mesh_t& mesh) {

	MPI_Comm_dup(comm, &(grid.comm));

	MPI_Comm_rank(comm, &(grid.rank));
	MPI_Comm_size(comm, &(grid.size));

	int canPrint = (grid.rank == 0);
	int argIdx;

	// If help is requested, all other arguments will be ignored
	if ((FindAndClearArgument("-h", argc, argv) != -1) || (FindAndClearArgument("--help", argc, argv) != -1))	{
		if (canPrint)
			PrintUsage(argv[0]);
		MPI_Abort(comm, STATUS_ERR);
	}

	// Topology information must always be present
	argIdx = FindAndClearArgument("-g", argc, argv);
	if (argIdx == -1)	{
		OneErrPrintf(canPrint, "Error: Could not find the topology information.\n");
		MPI_Abort(comm, STATUS_ERR);
	}	else {
		grid.Ncol = ExtractNumber(argIdx + 1, argc, argv);
		grid.Nrow = ExtractNumber(argIdx + 2, argc, argv);

		// At least the first topology dimension must be specified
		if (grid.Ncol <= 0){
			OneErrPrintf(canPrint, "Error: The topology size is invalid (first value: %d)\n", grid.Ncol);
			MPI_Abort(comm, STATUS_ERR);
		}

		// If the second topology dimension is missing, it will default to 1
		if (grid.Nrow <= 0){
			grid.Nrow = 1;
		}
	}

	// The domain size information is optional
	argIdx = FindAndClearArgument("-m", argc, argv);
	if (argIdx == -1)	{
		mesh.Nx = mesh.Ny = DEFAULT_DOMAIN_SIZE;
	}	else {
		mesh.Nx = ExtractNumber(argIdx + 1, argc, argv);
		mesh.Ny = ExtractNumber(argIdx + 2, argc, argv);

		// At least the first domain dimension must be specified
		if (mesh.Nx < MIN_DOM_SIZE) {
			OneErrPrintf(canPrint, "Error: The local domain size must be at least %d (currently: %d)\n", MIN_DOM_SIZE, mesh.Nx);
			MPI_Abort(comm, STATUS_ERR);
		}

		// If the second domain dimension is missing, it will default to the first dimension's value
		if (mesh.Ny <= 0) {
			mesh.Ny = mesh.Nx;
		}
	}

	// At the end, there should be no other arguments that haven't been parsed
	for (int i = 1; i < argc; ++i) {
		if (strlen(argv[i]) > 0) {
			OneErrPrintf(canPrint, "Error: Unknown argument (\"%s\")\n", argv[i]);
			MPI_Abort(comm, STATUS_ERR);
		}
	}

	// If we reach this point, all arguments were parsed successfully
	if (canPrint)	{
		printf("Topology size: %d x %d\n", grid.Ncol, grid.Nrow);
		printf("Local domain size (current node): %d x %d\n", mesh.Nx, mesh.Ny);
		printf("Global domain size (all nodes): %d x %d\n", grid.Ncol * mesh.Nx, grid.Nrow * mesh.Ny);
	}
}
