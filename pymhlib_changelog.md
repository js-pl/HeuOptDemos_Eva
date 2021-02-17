## Changes to existing pymhlib files

### log.py

- an additional parser argument **mh_log_step** was added to set a logfile path which is used for logging step-wise information necessary for visualization
- **init_logger()**: added initialization of the step logger, if no filepath was given, step-wise logging is omitted

### scheduler.py

- **Scheduler.__init__()**: added attribute for step logger
- **Scheduler.perform_method()**: added logging of detailed information about current solution before and after the scheduler method is performed, but only if the step logger is activated (i.e. the logger has handlers)


### solution.py

- added parser arguments **mh_grc_k, mh_grc_alpha, mh_grc_par** to pass information about parameters used in greedy randomized construction
- **Solution.__init__()**: added attribute for step logger
- **Solution** class: 
    - added methods for performing GRASP
        - **greedy_randomized_construction()**: performs all steps of a greedy randomized construction, also includes step-wise logging
        - **restricted_candidate_list()**: returns an array of solution elements selected from the candidate list, calls selection functions according to the setting parameters for greedy randomized construction
        - *abstract* methods which are called by greedy_randomized_construction() and restricted_candidate_list(), they have to be overwritten in specific implementations of solution classes
            - **copy_empty()**: returns a copy of the given solution but with an empty solution attribute
            - **is_complete_solution()**: returns *True* if the given solution is complete, *False* otherwise
            - **candidate_list()**: returns a dictionary, keys = solution elements, values = value of solution element if it is chosen to be added to the solution
            - **update_solution()**: adds the element which was randomly selected from the restricted candidate list to the solution
            - **restricted_candidate_list_k()**: selects k best elements from a candidate list
            - **restricted_candidate_list_alpha()**: selects elements according to some parameter alpha
    - **construct_greedy()**: implements a greedy construction heuristic which can be used to obtain an initial solution
    - *abstract* **is_tabu()**: returns True if a solution is forbidden according to givent tabu list, solution specific implementation is necessary
    - *abstract* **get_tabu_attribute()**: given an old and a new solution this method returns the attribute which is tabu for some iterations, solution specific implementation is necessary


### binvec_solution.py
- implementation of GRASP methods in **BinaryVectorSolution** class
	 - update_solution(), copy_empty(), is_complete_solution(), candidate_list()
- implemtation of methods for Tabu Search
    - **is_tabu()**
    - **get_tabu_attribute()**


### subsetvec_solution.py
- implementation of GRASP methods in **SubsetVectorSolution** class
	- update_solution(), copy_empty(), is_complete_solution(), candidate_list()
- implemtation of methods for Tabu Search
    - **is_tabu()**
    - **get_tabu_attribute()**


### demos.maxsat.py
- **MAXSATSolution**: implemented problem specific methods for GRASP
    - restricted_candidate_list_k(), restricted_candidate_list_alpha()


### demos.misp.py
- **MISPSolution**: implemented problem specific methods for GRASP
    - restricted_candidate_list_k(), restricted_candidate_list_alpha()



## New pymhlib files

### ts.py
implementation of Tabu Search

### ts_helper.py
implementation of helper classes 
- **TabuList**: manages a tabu list (currently implemented as simple list) which holds TabuAttributes
- **TabuAttribute**: holds a tabu attribute and its current tabu status (i.e. lifespan)