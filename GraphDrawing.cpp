#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <limits>
#include <chrono>

const int BOARD_SIZE = 700;
const bool VERBOSE_MODE = false;

using namespace std;


class Vertex {
    public:
        int x;
        int y;
        Vertex() {}
        Vertex(int tx, int ty): x(tx), y(ty) {}
        Vertex(const Vertex &v): x(v.x), y(v.y) {}
};


class Edge {
    public:
        int vo; // origin
        int vd; // destination
        double w; // weight
        Edge() {}
        Edge(int to, int td, double tw) : vo(to), vd(td), w(tw) {} 
};


void print_adjacency_list(vector<vector<Edge>> &al, bool verbose_mode = VERBOSE_MODE) {

    if (verbose_mode == true) {
        fprintf(stderr, "Printing adjacency list...\n");
        for(int i = 0; i < al.size(); i++) {
            fprintf(stderr, "Vertex %d: ", i);
            for(int j = 0; j < al[i].size(); j++) {
                fprintf(stderr, "%d -> %d (%f); ", al[i][j].vo, al[i][j].vd, al[i][j].w);
            }
            fprintf(stderr, "\n");
        }
    } else {}
}


void print_matrix(vector<vector<double>> &mr, bool verbose_mode = VERBOSE_MODE) {

    if (verbose_mode == true) {
        fprintf(stderr, "Printing matrix...\n");
        for(int i = 0; i < mr.size(); i++) {
            for(int j = 0; j < mr[i].size(); j++) {
                fprintf(stderr, "%4.2f ", mr[i][j]);
            }
            fprintf(stderr, "\n");
        }
    } else {}
}

double distance(Vertex &vo, Vertex &vd) {

    // Get distance between two vertexes.

    int vox = vo.x;
    int voy = vo.y;

    int vdx = vd.x;
    int vdy = vd.y;

    double dx2 = (vox - vdx)*(vox - vdx);
    double dy2 = (voy - vdy)*(voy - vdy);

    double d = sqrt( dx2 + dy2 );
    //double d = abs(vox - vdx) + abs(voy - vdy);
    return d;
}


void base_adjacency_list_rep(vector<vector<Edge>> &bal, vector<int> &edges) {

    // Construct a base adjacency list (bal) representation of a graph.
    // This function is used only for the initial graph i. e.
    // it uses the provided number of vertexes (N) and the
    // provided edges (edges).
    // The constructed bal is used as a reference 
    // throughout the entire solution.

    for(int i = 0; i < edges.size(); i = i + 3) {
        int vo = edges[i + 0];
        int vd = edges[i + 1];
        double w = edges[i + 2];        
        bal[vo].push_back(Edge(vo, vd, w));
        bal[vd].push_back(Edge(vd, vo, w));
    }

}


void adjustable_adjacency_list_rep(vector<vector<Edge>> &aal, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    // Construct a adjustable adjacency list (aal) representation of a graph.
    // In our solution we will try to adjust the aal so it resembles
    // bal as close as possible. 

    for(int i = 0; i < bal.size(); i++) {
        for(int j = 0; j < bal[i].size(); j++) {
            int vo = bal[i][j].vo;
            int vd = bal[i][j].vd;

            double w = distance(vp[vo], vp[vd]);
            aal[i].push_back(Edge(vo, vd, w));
        }
    }

}


void base_matrix_rep(vector<vector<double>> &bmr, vector<int> &edges) {

    // Construct a base matrix representation of a graph.
    // This function is used only for the initial graph i. e.
    // it uses the provided number of vertexes (N) and the
    // provided edges (edges).

    for(int i = 0; i < edges.size(); i = i + 3){
        int vo = edges[i + 0];
        int vd = edges[i + 1];
        double w = edges[i + 2];

        bmr[vo][vd] = w;
        bmr[vd][vo] = w;
    }

}


void adjustable_matrix_rep(vector<vector<double>> &amr, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    // Construct a adjustable matrix representation (amr) of a graph.
    // In our solution we will try to adjust the amr so it resembles
    // bmr as close as possible.
    // bal is necessary because we need to know which values of the
    // matrix to fill.

    for(int i = 0; i < bal.size(); i++) {
        for(int j = 0; j < bal[i].size(); j++) {
            int vo = bal[i][j].vo;
            int vd = bal[i][j].vd;

            double w = distance(vp[vo], vp[vd]);
            amr[vo][vd] = w;
        }
    }

}


void update_vertex_adjustable_adjacency_list(int &vi, vector<vector<Edge>> &aal, vector<Vertex> &vp) {

    // update performed for a single vertex!

    // This function assumes that the state of vp has changed.
    // The position of one vertex (vi) has changed in vp.
    // The corresponding weights in aal have to be updated.
    // This function performs the required update.

    for(int i = 0; i < aal[vi].size(); i++) {
        int vo = aal[vi][i].vo;
        int vd = aal[vi][i].vd;
        
        double d = distance(vp[vo], vp[vd]);
        aal[vi][i].w = d;

        for(int j = 0; j < aal[vd].size(); j++) {
            if (aal[vd][j].vd != vo)
                continue;

            int vdr = aal[vd][j].vd;

            double dr = distance(vp[vd], vp[vdr]);
            aal[vd][j].w = dr;
        } // inner for end

    } // outer for end
}


void update_adjustable_adjacency_list(vector<int> vua, vector<vector<Edge>> &aal, vector<Vertex> &vp) {

    // This function assumes that the state of vp has changed.
    // The positions of vertexes given in the vertex update array (vua)
    // have been changed.
    // The corresponding weights in aal have to be updated.
    // This function performs the required update.

    for(int i = 0; i < vua.size(); i++) 
        update_vertex_adjustable_adjacency_list(vua[i], aal, vp);

}


void update_vertex_matrix_rep(int &vi, vector<vector<double>> &amr, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    // update performed for a single vertex!

    // This function assumes that the state of vp has changed.
    // The position of one vertex (vi) has changed in vp.
    // The corresponding weights in amr have to be updated.
    // This function performs the required update.
    // In its operation it is analogous to
    // update_vertex_adjustable_adjacency_list.

    for(int i = 0; i < bal[vi].size(); i++) {
        int vo = bal[vi][i].vo;
        int vd = bal[vi][i].vd;

        double d = distance(vp[vo], vp[vd]);
        amr[vo][vd] = d;
        amr[vd][vo] = d;
    }
}


void update_matrix_rep(vector<int> vua, vector<vector<double>> &amr, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    // This function assumes that the state of vp has changed.
    // The positions of vertexes given in the vertex update array (vua)
    // have been changed.
    // The corresponding weights in amr have to be updated.
    // This function performs the required update.
    // In its operation it is analogous to
    // update_adjustable_adjacency_list.

    for(int i = 0; i < vua.size(); i++) 
        update_vertex_matrix_rep(vua[i], amr, bal, vp);

}


void update_ratios(int &vi, double &min_ratio, double &max_ratio, vector<vector<double>> &amr, vector<vector<Edge>> &bal) {

    //Updates the min_ratio and max_ratio for a given vi vertex.

    // Loop over edges of the vi vertex.
    for(int j = 0; j < bal[vi].size(); j++) {
            int vo = bal[vi][j].vo;
            int vd = bal[vi][j].vd;

            double desired_w = bal[vi][j].w;
            double current_w = amr[vo][vd];
            double ratio = current_w/desired_w;

            if (min_ratio > ratio)
                min_ratio = ratio;
            if (max_ratio < ratio)
                max_ratio = ratio;
    }
}


void update_ratios_pass(int &vi, 
                        double &min_ratio, 
                        int &min_x,
                        int &min_y,
                        double &max_ratio,
                        int &max_x,
                        int &max_y,
                        vector<vector<double>> &amr, 
                        vector<vector<Edge>> &bal) {

    //Updates the min_ratio and max_ratio for a given vi vertex.

    // Loop over edges of the vi vertex.
    for(int j = 0; j < bal[vi].size(); j++) {
            int vo = bal[vi][j].vo;
            int vd = bal[vi][j].vd;

            double desired_w = bal[vi][j].w;
            double current_w = amr[vo][vd];
            double ratio = current_w/desired_w;

            if (min_ratio > ratio) {
                min_ratio = ratio;
                min_x = vo;
                min_y = vd;
            }
            if (max_ratio < ratio) {
                max_ratio = ratio;
                max_x = vo;
                max_y = vd;
            }

    }
}


void update_ratios_pass_global_detect(int &vi, 
                                      double &min_ratio, 
                                      int &min_x,
                                      int &min_y,
                                      int &old_min_x,
                                      int &old_min_y,
                                      double &max_ratio,
                                      int &max_x,
                                      int &max_y,
                                      int &old_max_x,
                                      int &old_max_y,
                                      vector<vector<double>> &amr, 
                                      vector<vector<Edge>> &bal,
                                      bool &min_is_global,
                                      bool &max_is_global) {

    //Updates the min_ratio and max_ratio for a given vi vertex.

    // Loop over edges of the vi vertex.
    for(int j = 0; j < bal[vi].size(); j++) {
            int vo = bal[vi][j].vo;
            int vd = bal[vi][j].vd;

            double desired_w = bal[vi][j].w;
            double current_w = amr[vo][vd];
            double ratio = current_w/desired_w;

            if ((old_min_x == vo) && (old_min_y == vd))
                min_is_global = false;

            if ((old_max_x == vo) && (old_max_y == vd))
                max_is_global = false;

            if (min_ratio > ratio) {
                min_ratio = ratio;
                min_x = vo;
                min_y = vd;
            }

            if (max_ratio < ratio) {
                max_ratio = ratio;
                max_x = vo;
                max_y = vd;
            }

    }
}


pair<double, double> calculate_score(vector<vector<double>> &amr, vector<vector<Edge>> &bal) {

    // Calculates the overall score given am amr.
    
    double min_ratio = numeric_limits<double>::max();
    double max_ratio = -1.0*numeric_limits<double>::max();

    // Loop over vertexes.
    for(int i = 0; i < bal.size(); i++)
        update_ratios(i, min_ratio, max_ratio, amr, bal);

    return make_pair(min_ratio, max_ratio);
}


double calculate_score_ratio_pass(double &min_ratio, 
                                  int &min_x,
                                  int &min_y,
                                  double &max_ratio,
                                  int &max_x,
                                  int &max_y, 
                                  vector<vector<double>> &amr, 
                                  vector<vector<Edge>> &bal) {

    // Calculates the overall score given am amr.

    // Loop over vertexes.
    for(int i = 0; i < bal.size(); i++)
        update_ratios_pass(i, min_ratio, min_x, min_y, max_ratio, max_x, max_y, amr, bal);

    return min_ratio/max_ratio;
}


void initialize_vertex_positions(vector<Vertex> &vp, vector<vector<bool>> &vm) {

    // Assign random positions to the vertexes.
    // Mark the used positions in vm.

    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<> uid(0, BOARD_SIZE - 1);

    for(int i = 0; i < vp.size(); i++) {
        while(true) {        
            int x = uid(g);
            int y = uid(g); 

            // Two vertexes cannot occupy the same possition.
            // Check if the drawn points are not occupied.
            if ((vm[x][y] == true) || (vm[y][x] == true))               
                continue;

            vm[x][y] = true;
            vm[y][x] = true;

            vp[i].x = x;
            vp[i].y = y;
            break;

        } // while end
    } // for end
}


bool approve_visit(int &ox, int &oy, int &nx, int &ny, vector<vector<bool>> &vm) {

    // Check is it is possible to move a vertex with coordinates (ox, oy) 
    // to a new position (nx, ny).
    // If yes update the vm matrix.

    if (vm[nx][ny] == true || vm[ny][nx] == true)
        return false;

    vm[ox][oy] = false;
    vm[oy][ox] = false;

    vm[nx][ny] = true;
    vm[ny][nx] = true;    

    return true;
}


bool random_vertex_position_update(Vertex &v, vector<vector<bool>> &vm, int &lb, int &rb) {

    // Move vertex to a new position.
    // The move is performed by drawing a random shift
    // from [lb, rb] X [lb, rb].

    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<> uid(lb, rb);
    
    int dx = uid(g);
    int dy = uid(g);

    if ((dx == 0) && (dy == 0))
        return false;

    int nx = v.x + dx;
    int ny = v.y + dy;

    if ((nx > BOARD_SIZE - 1) || (nx < 0))
        return false;

    if ((ny > BOARD_SIZE - 1) || (ny < 0))
        return false;
    
    bool apv = approve_visit(v.x, v.y, nx, ny, vm);

    if (apv == true) {
        v.x = nx;
        v.y = ny;
        return true;
    } else
        return false;
}


bool random_vertex_teleport(Vertex &v, vector<vector<bool>> &vm) {

    // Moves the vertex to a random position on the board.

    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<> uid(0, BOARD_SIZE - 1);
    
    int nx = uid(g);
    int ny = uid(g);
    
    bool apv = approve_visit(v.x, v.y, nx, ny, vm);

    if (apv == true) {
        v.x = nx;
        v.y = ny;
        return true;
    } else
        return false;

}


void random_vertex_move(int &vi, 
                 vector<vector<double>> &amr,
                 vector<vector<Edge>> &bal, 
                 vector<Vertex> &vp,
                 vector<vector<bool>> &vm,
                 int &lb,
                 int &rb) {

    // Move vertex vi to a new position within the (BOARD_SIZE x BOARD_SIZE) grid.
    // When the vertex is moved the following need to be updated:
    // - the adjustable matrix representation (amr)
    // - the adjustable adjacency list (aal)
    // - the min_ratio
    // - the max_ratio
    // At the moment the aal is not used an so it is not updated.

    bool made_move = false;
    made_move = random_vertex_position_update(vp[vi], vm, lb, rb);

    if (made_move == true) {
        update_vertex_matrix_rep(vi, amr, bal, vp);
        //update_ratios(vi, amr, bal);
    }

}


bool deterministic_vertex_position_update(Vertex &v, int &nx, int &ny, vector<vector<bool>> &vm) {

    // Move vertex to a new position given by (nx, ny).

    if ((vm[nx][ny] == true) || (vm[ny][nx] == true))
        return false;

    v.x = nx;
    v.y = ny;

    return true;
}


void deterministic_vertex_move(int &vi,
                               int &nx,
                               int &ny, 
                               vector<vector<double>> &amr,
                               vector<vector<Edge>> &bal, 
                               vector<Vertex> &vp,
                               vector<vector<bool>> &vm) {

    // Moves vertex vi to a new position given by (x, y).
    // The adjustable matrix representation (amr) is updated.

    deterministic_vertex_position_update(vp[vi], nx, ny, vm);
    update_vertex_matrix_rep(vi, amr, bal, vp);
}


inline double metropolis_ratio(double &ns, double &os, double &T) {

    // ns - new score
    // os - old score
    // T - temperature

    return exp( (ns - os) / T );
}


void adjust_min_ratio(double &old_min_ratio, 
                      int &old_min_x, 
                      int &old_min_y, 
                      double &min_ratio, 
                      int &min_x, 
                      int &min_y,
                      bool &min_is_global) {

    if (min_ratio <= old_min_ratio) {
        //fprintf(stderr, "min one\n");        
        return;
    }

    if ((min_ratio > old_min_ratio) && (min_is_global == true)) {
        //fprintf(stderr, "min two\n");  
        min_ratio = old_min_ratio;
        min_x = old_min_x;
        min_y = old_min_y;
        return;
    }

    if ((min_ratio > old_min_ratio) && (min_is_global == false)) {
        //fprintf(stderr, "min three\n");  
        return;
    }

}


void adjust_max_ratio(double &old_max_ratio, 
                      int &old_max_x, 
                      int &old_max_y, 
                      double &max_ratio, 
                      int &max_x, 
                      int &max_y,
                      bool &max_is_global) {

    if (max_ratio >= old_max_ratio) {
        //fprintf(stderr, "max one\n");      
        return;
    }

    if ((max_ratio < old_max_ratio) && (max_is_global == true)) {
        //fprintf(stderr, "max two\n");
        max_ratio = old_max_ratio;
        max_x = old_max_x;
        max_y = old_max_y;
        return;
    }

    if ((max_ratio < old_max_ratio) && (max_is_global == false)) {
        //fprintf(stderr, "max three\n");
        return;
    }

}


vector<Vertex> sa(vector<Vertex> &vp, 
                  vector<vector<double>> &amr, 
                  vector<vector<Edge>> &bal,
                  vector<vector<bool>> &vm,
                  int &lb,
                  int &rb) {
    
    int N = vp.size();
    vector<Vertex> optimal_solution(vp); // optimal solution

    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<> choose_vertex(0, N - 1);
    uniform_int_distribution<> choose_grid_coordinate(0, N - 1);
    uniform_real_distribution<> uniform(0.0, 1.0);

    double T = 20.0; // temperature
    double tT = 0.1; // termination temperature
    double tdr = 0.9; // temperature decrease rate    
    double nI = 100000; // number of iterations per temperature step

    pair<double, double> p_os = calculate_score(amr, bal); // old score
    double os = p_os.first/p_os.second;

    double old_min_ratio = numeric_limits<double>::max();
    double old_max_ratio = -1.0*numeric_limits<double>::max();
    int old_min_x = -1;
    int old_min_y = -1;
    int old_max_x = -1;
    int old_max_y = -1;

    double os_t = calculate_score_ratio_pass(old_min_ratio, old_min_x, old_min_y, old_max_ratio, old_max_x, old_max_y, amr, bal); 

    double maximal_score = 0.0;

    //fprintf(stderr, "maximal_score = %f\n", maximal_score);
    while(T > tT) {

        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

        for(int i = 0; i < nI; i++) {

            int vi = choose_vertex(g);            
            int ox = vp[vi].x;
            int oy = vp[vi].y;
            bool made_move = random_vertex_position_update(vp[vi], vm, lb, rb);

            if (made_move == true)            
                update_vertex_matrix_rep(vi, amr, bal, vp);
            else
                continue;

            double min_ratio = numeric_limits<double>::max();
            double max_ratio = -1.0*numeric_limits<double>::max();
            int min_x = old_min_x;
            int min_y = old_min_y;
            int max_x = old_max_x;
            int max_y = old_max_y;
            bool min_is_global = true;
            bool max_is_global = true;

            update_ratios_pass_global_detect(vi, 
                                             min_ratio, 
                                             min_x,
                                             min_y,
                                             old_min_x,
                                             old_min_y,
                                             max_ratio,
                                             max_x,
                                             max_y,
                                             old_max_x,
                                             old_max_y,
                                             amr, 
                                             bal,
                                             min_is_global,
                                             max_is_global);

            //fprintf(stderr, "=========================================================\n");
            //fprintf(stderr, "  old           min_ratio: %f max_ratio: %f\n", old_min_ratio, old_max_ratio);
            //fprintf(stderr, "  pre new       min_ratio: %f max_ratio: %f\n", min_ratio, max_ratio);

            adjust_min_ratio(old_min_ratio, 
                             old_min_x, 
                             old_min_y, 
                             min_ratio, 
                             min_x, 
                             min_y,
                             min_is_global);

            adjust_max_ratio(old_max_ratio, 
                             old_max_x, 
                             old_max_y, 
                             max_ratio, 
                             max_x, 
                             max_y,
                             max_is_global);


            //double ns = calculate_score(amr, bal);


            //pair<double, double> p_ns = calculate_score(amr, bal);
            //double ns = p_ns.first/p_ns.second;

            double ns_t = min_ratio/max_ratio;
            //fprintf(stderr, "ns   = %f min_ratio: %f max_ratio: %f\n", ns, p_ns.first, p_ns.second);
            //fprintf(stderr, "ns_t = %f min_ratio: %f max_ratio: %f\n", ns_t, min_ratio, max_ratio);


            double p = metropolis_ratio(ns_t, os_t, T);

            if ( p > uniform(g) ) {

                //os = ns;
                
                old_min_ratio = min_ratio;
                old_max_ratio = max_ratio;
                old_min_x = min_x;
                old_min_y = min_y;
                old_max_x = max_x;
                old_max_y = max_y;

                os_t = old_min_ratio/old_max_ratio;

                // Update optimal solution                 
                if ( maximal_score < ns_t ) {
                    maximal_score = ns_t;
                    optimal_solution = vp;
                }
     
            } else {
                // New state not accepted.
                // Revert back to old state.

                // Revert true/false in vm.
                approve_visit(vp[vi].x, vp[vi].y, ox, oy, vm);
                vp[vi].x = ox;
                vp[vi].y = oy;
                update_vertex_matrix_rep(vi, amr, bal, vp);


            } // outer if end

        } // for loop end


        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        int elapsed_time = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();        

        fprintf(stderr, "---->   ===   <----");
        fprintf(stderr, "T = %f\n", T);
        fprintf(stderr, "maximal_score = %f\n", maximal_score);
        fprintf(stderr, "Time: %d ms\n", elapsed_time);

        T = T*tdr;
    } // while loop end

    return optimal_solution;

}


void dispatch_vertex_positions(vector<int> &ret, vector<Vertex> &vp) {

    // Converts the custom vertex positions representation into
    // the one required by the competition i. e. vector of ints.

    for(int i = 0; i < vp.size(); i++) {
        ret[2*i + 0] = vp[i].x;
        ret[2*i + 1] = vp[i].y;
    }

}


class GraphDrawing {
    public:
        vector<int> plot(int N, vector<int> edges) {

            // bmr - base matrix representation of the graph
            // This is the matrix representation that we are trying
            // to achive.
            vector<vector<double>> bmr(N, vector<double>(N, 0.0));

            base_matrix_rep(bmr, edges);
            print_matrix(bmr);


            //vp - vertex positions
            vector<Vertex> vp(N);

            // vm - visit matrix
            // Marks the points that have been visited so far.
            vector<vector<bool>> vm(BOARD_SIZE, vector<bool>(BOARD_SIZE, false));

            // initialize the vertex positions with some
            // random values
            initialize_vertex_positions(vp, vm);

            vector<vector<Edge>> bal(N, vector<Edge>(0));
            base_adjacency_list_rep(bal, edges);
            print_adjacency_list(bal);

            //Adjustable matrix
            vector<vector<double>> amr(N, vector<double>(N, 0.0));
            adjustable_matrix_rep(amr, bal, vp);
            print_matrix(amr);

            //Adjustable adjacency list
            vector<vector<Edge>> aal(N, vector<Edge>(0));
            adjustable_adjacency_list_rep(aal, bal, vp);
            print_adjacency_list(aal);


            /*
            // Changeing the position of vertex 3.
            int index = 4;
            vp[index].x = 100;
            vp[index].y = 200;
            //update_vertex_adjustable_adjacency_list(index, aal, vp);
            //print_adjacency_list(aal);
            update_vertex_matrix_rep(index, amr, bal, vp);
            print_matrix(amr);


            vector<int> vua = {0, 1};
            vp[vua[0]].x = 300;
            vp[vua[0]].y = 300;
            vp[vua[1]].x = 400;
            vp[vua[1]].y = 400;
            //update_adjustable_adjacency_list(vua, aal, vp);
            //print_adjacency_list(aal);
            update_matrix_rep(vua, amr, bal, vp);
            print_matrix(amr);

            double score = calculate_score(amr, bal);
            fprintf(stderr, "-----\n"); 
            fprintf(stderr, "Score = %4.16f\n", score);
            fprintf(stderr, "Score = %4.16f\n", score);
            fprintf(stderr, "Score = %4.16f\n", score);

            index = 3;
            random_vertex_move(index, amr, bal, vp, vm);
            score = calculate_score(amr, bal);
            fprintf(stderr, "-----\n"); 
            fprintf(stderr, "Score = %4.16f\n", score);
            fprintf(stderr, "Score = %4.16f\n", score);
            fprintf(stderr, "Score = %4.16f\n", score);

            int nx = 699;
            int ny = 699;
            deterministic_vertex_move(index, nx, ny, amr, bal, vp, vm); 
            score = calculate_score(amr, bal);
            fprintf(stderr, "-----\n");            
            fprintf(stderr, "Score = %4.16f\n", score);
            fprintf(stderr, "Score = %4.16f\n", score);
            fprintf(stderr, "Score = %4.16f\n", score);
            */

            int lb = -10;
            int rb = 10;
            vector<Vertex> optimal_solution = sa(vp, amr, bal, vm, lb, rb);

            vector<int> ret(2*N);
            dispatch_vertex_positions(ret, optimal_solution);
            return ret;
        }
};
// -------8<------- end of solution submitted to the website -------8<-------

template<class T> void getVector(vector<T>& v) {
    for (int i = 0; i < v.size(); ++i)
        cin >> v[i];
}

int main() {
    GraphDrawing gd;
    int N;
    cin >> N;
    int E;
    cin >> E;
    vector<int> edges(E);
    getVector(edges);
    
    vector<int> ret = gd.plot(N, edges);
    cout << ret.size() << endl;
    for (int i = 0; i < (int)ret.size(); ++i)
        cout << ret[i] << endl;
    cout.flush();
}
