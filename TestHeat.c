/* Solving the Laplace Equation in 2D
	ut = uxx + uyy
	with Dirchlet BC
	u = 0 on each boundary
	u = sin(pi x) sin(pi y) intitally
*/

static char help[] = "Solves 2D Heat";

# include <petscts.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **args)
{
	Vec				u; 		/* Initial cond, solution */
	Mat				D; 		/* DE matrix */
	TS				ts; 	/* DAE solver */
	PetscErrorCode	ierr;
	PetscMPIInt		size;
	PetscInt 		i, j, iI, col[3], n = 7, m = 5;
	PetscScalar		value[3], hx, u0value, tf = 1.0, one = 1.0, dt;
	
	PetscInitialize(&argc, &args, (char*)0, help);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL); CHKERRQ(ierr);
	m = n - 2;
	hx = 1.0 / (n - 1.0);
	one = 1.0 / (hx*hx);
	dt = hx*hx / 4.0;
	
	/*
		--------------------------------------------------
		Create the matrix, RHS, and solution vector
		--------------------------------------------------
	*/
	
	ierr = VecCreate(PETSC_COMM_WORLD, &u); CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) u, "Initial/Soln"); CHKERRQ(ierr);
	ierr = VecSetSizes(u, PETSC_DECIDE, m*m); CHKERRQ(ierr);
	ierr = VecSetFromOptions(u); CHKERRQ(ierr);
	
	ierr = MatCreate(PETSC_COMM_WORLD, &D); CHKERRQ(ierr);
	ierr = MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, m*m, m*m); CHKERRQ(ierr);
	ierr = MatSetFromOptions(D); CHKERRQ(ierr);
	ierr = MatSetUp(D); CHKERRQ(ierr);
	
	/* Set Matrix Values */
	for (i = 0; i < m*m; i++) {
		if ((i % m) == 0) {
			value[0] = -4.0/(hx*hx); value[1] = 1.0/(hx*hx);
			col[0] = i; col[1] = i + 1;
			ierr = MatSetValues(D, 1, &i, 2, col, value, INSERT_VALUES); CHKERRQ(ierr);
		}
		else if ((i % m) == m - 1) {
			value[0] = 1.0/(hx*hx); value[1] = -4.0/(hx*hx);
			col[0] = i - 1; col[1] = i;
			ierr = MatSetValues(D, 1, &i, 2, col, value, INSERT_VALUES); CHKERRQ(ierr);
		}
		else {
			value[0] = 1.0/(hx*hx); value[1] = -4.0/(hx*hx); value[2] = 1.0/(hx*hx);
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
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			u0value = PetscSinReal(PETSC_PI * (i + 1) / (n - 1))*PetscSinReal(PETSC_PI * (j + 1) / (n - 1));
			iI = i + m*j;
			ierr = VecSetValues(u, 1, &iI, &u0value, INSERT_VALUES); CHKERRQ(ierr);
		}
	}
	ierr = VecAssemblyBegin(u); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(u); CHKERRQ(ierr);
	
	/*
		--------------------------------------------------
		Create the DAE solver
		--------------------------------------------------
	*/
	
	ierr = TSCreate(PETSC_COMM_WORLD, &ts); CHKERRQ(ierr);
	ierr = TSSetType(ts, TSEULER); CHKERRQ(ierr);
	ierr = TSSetInitialTimeStep(ts, 0.0, dt); CHKERRQ(ierr);
	ierr = TSSetExactFinalTime(ts, tf); CHKERRQ(ierr);
	ierr = TSSetRHSJacobian(ts, D, D, TSComputeRHSJacobianConstant, NULL); CHKERRQ(ierr);
	ierr = TSSetRHSFunction(ts, NULL, TSComputeRHSFunctionLinear, NULL); CHKERRQ(ierr);
	
	/*
		--------------------------------------------------
		Solve the DAE system
		--------------------------------------------------
	*/
	
	ierr = TSSolve(ts, u); CHKERRQ(ierr);
	ierr = TSView(ts, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	ierr = MatView(D, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	/*
		--------------------------------------------------
		Clean up
		--------------------------------------------------
	*/
	
	ierr = VecDestroy(&u); CHKERRQ(ierr);
	ierr = MatDestroy(&D); CHKERRQ(ierr);
	ierr = TSDestroy(&ts); CHKERRQ(ierr);
	
	ierr = PetscFinalize();
	return 0;
}