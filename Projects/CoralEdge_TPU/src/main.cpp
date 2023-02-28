#include <stdio.h>
#include "model.h"

//#define CREATE_MODEL
#define DRIVE_MODEL


int main(int argc, char* argv[])
{
    //printf("TfLiteVersion : %s\n", TfLiteVersion());
#ifdef CREATE_MODEL
	create_model();
#endif

#ifdef DRIVE_MODEL
	drive_model();
#endif

	return 0;
}