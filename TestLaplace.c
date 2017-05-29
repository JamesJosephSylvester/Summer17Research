/* Solving the Laplace Equation in 2D
	uxx + uyy = f, f = 0
	with Dirchlet BC
	u = sin(pi t) on each boundary
*/

static char help[] = "Solves 2D Laplace";

# include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **args)
{
	Vec				f, u; 	/* RHS, solution */
	Mat				D; 		/* DE matrix */
	KSP				ksp; 	/* linear solver */
	PetscErrorCode	ierr;
	PetscMPIInt		size;
	PetscInt 		i, j, col[3], n = 7, m = 5;
	PetscScalar		value[3], bdvalue, one = -1.0;
	
	PetscInitialize(&argc, &args, (char*)0, help);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);
	m = n - 2;
	one = 1.0/(hx*hx);
	/*
		--------------------------------------------------
		Create the matrix, RHS, and solution vector
		--------------------------------------------------
	*/
	
	ierr = VecCreate(PETSC_COMM_WORLD, &f); CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) f, "Right Hand Side"); CHKERRQ(ierr);
	ierr = VecSetSizes(f, PETSC_DECIDE, m*m); CHKERRQ(ierr);
	ierr = VecSetFromOptions(f); CHKERRQ(ierr);
	ierr = VecDuplicate(f, &u); CHKERRQ(ierr);
	
	ierr = MatCreate(PETSC_COMM_WORLD, &D); CHKERRQ(ierr);
	ierr = MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, m*m, m*m); CHKERRQ(ierr);
	ierr = MatSetFromOptions(D); CHKERRQ(ierr);
	ierr = MatSetUp(D); CHKERRQ(ierr);
	
	/* Set Matrix Values */
	for (i = 0; i < m*m; i++) {
		if ((i % m) == 0) {
			value[0] = 4.0; value[1] = -1.0;
			col[0] = i; col[1] = i + 1;
			ierr = MatSetValues(D, 1, &i, 2, col, value, INSERT_VALUES); CHKERRQ(ierr);
		}
		else if ((i % m) == m - 1) {
			value[0] = -1.0; value[1] = 4.0;
			col[0] = i - 1; col[1] = i;
			ierr = MatSetValues(D, 1, &i, 2, col, value, INSERT_VALUES); CHKERRQ(ierr);
		}
		else {
			value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
			col[0] = i - 1; col[1] = i; col[2] = i + 1;
			ierr = MatSetValues(D, 1, &i, 3, col, value, INSERT_VALUES); CHKERRQ(ierr);
		}
		if (i <= m*m - 1 - m) {
			j = i + m;
			ierr = MatSetValues(D, 1, &i, 1, &j, &one, INSERT_VALUES); CHKERRQ(ierr);
			}
		if (i >= m) {
			j = i - m;
			ierr = MatSetValues(D, 1, &i, 1, &j, &one, INSERT_VALUES); CHKERRQ(ierr);
			}
		}
	ierr = MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	
	/* Set Vectors */
	ierr = VecSet(u, 0.0); CHKERRQ(ierr);
	ierr = VecSet(f, 0.0); CHKERRQ(ierr);
	for (i = 0; i < m; i++) {
		bdvalue = PetscSinReal(PETSC_PI * (i + 1.) / (n - 1.));
		ierr = VecSetValues(f, 1, &i, &bdvalue, ADD_VALUES); CHKERRQ(ierr);
		j = i*m;
		ierr = VecSetValues(f, 1, &j, &bdvalue, ADD_VALUES); CHKERRQ(ierr);
		j = i + m*(m - 1);
		ierr = VecSetValues(f, 1, &j, &bdvalue, ADD_VALUES); CHKERRQ(ierr);
		j = i*m + m - 1;
		ierr = VecSetValues(f, 1, &j, &bdvalue, ADD_VALUES); CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(f); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(f); CHKERRQ(ierr);
	
	/*
		--------------------------------------------------
		Create the linear solver
		--------------------------------------------------
	*/
	
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp, D, D); CHKERRQ(ierr);
	
	/*
		--------------------------------------------------
		Solve the linear system
		--------------------------------------------------
	*/
	
	ierr = KSPSolve(ksp, f, u); CHKERRQ(ierr);
	ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = MatView(D, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = VecView(f, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	/*
		--------------------------------------------------
		Clean up
		--------------------------------------------------
	*/
	
	ierr = VecDestroy(&f); CHKERRQ(ierr);
	ierr = VecDestroy(&u); CHKERRQ(ierr);
	ierr = MatDestroy(&D); CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	
	ierr = PetscFinalize();
	return 0;
}