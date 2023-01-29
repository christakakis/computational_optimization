This parser was made to read TSP and more types of problem instances with the help of tsplib95.

Supported types are:

	1) Instances of 			TSP, HCP, ATSP, SOP, CVRP.
	
	2) Edge_Weight_Types of 		EXPLICIT, EUC_2D, EUC_3D, XRAY1, XRAY2, GEO, ATT.
	
	3) Edge_Weight_Formats of 		UPPER_ROW, LOWER_ROW, UPPER_COL, LOWER_COL, FULL_MATRIX.

The final tsp function created in this code returns the following information about the problem:

	1) Name
	
	2) Type
	
	3) Dimension
	
	4) Comment
	
	5) Distance
	
	6) Weight Matrix
	
	7) Capacity 		(if any)
	
	8) Depots 		(if any)
	
	9) Node Coordinates 	(if any)
	
Main goal of returning all those information about a selected problem is to apply heuristics and metaheuristics later.
