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
        int id;
        Edge() {}
        Edge(int to, int td, double tw, int tid = -1) : vo(to), vd(td), w(tw), id(tid) {} 
};


template<typename T> void print_vector(vector<T> vec) {

    cerr << "Printing array..." << endl;
    for(auto v : vec)
        cerr << v << endl;
}


void print_adjacency_list(vector<vector<Edge>> &al, bool verbose_mode = VERBOSE_MODE) {

    if (verbose_mode == true) {
        fprintf(stderr, "Printing adjacency list...\n");
        for(int i = 0; i < al.size(); i++) {
            fprintf(stderr, "Vertex %d: ", i);
            for(int j = 0; j < al[i].size(); j++) {
                fprintf(stderr, "%d -> %d (%f, %d); ", al[i][j].vo, al[i][j].vd, al[i][j].w, al[i][j].id);
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
    int id = 0;
    for(int i = 0; i < edges.size(); i = i + 3) {
        int vo = edges[i + 0];
        int vd = edges[i + 1];
        double w = edges[i + 2];        
        bal[vo].push_back(Edge(vo, vd, w, id));
        bal[vd].push_back(Edge(vd, vo, w, id));
        id++;
    }

}


void update_ratio_array(int &vi, vector<double> &ratio_array, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    for(int i = 0; i < bal[vi].size(); i++) {
        int vo = bal[vi][i].vo;
        int vd = bal[vi][i].vd;
        double w = bal[vi][i].w;
        int id = bal[vi][i].id;

        double ratio = distance(vp[vo], vp[vd])/w;
        ratio_array[id] = ratio;
    }
}


void fill_ratio_array(vector<double> &ratio_array, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    for(int vi = 0; vi < vp.size(); vi++) {
        update_ratio_array(vi, ratio_array, bal, vp);
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


double vertex_score(int &vi, vector<vector<double>> &amr, vector<vector<Edge>> &bal) {

    double ratio_sum = 0.0;

    for(int j = 0; j < bal[vi].size(); j++) {
            int vo = bal[vi][j].vo;
            int vd = bal[vi][j].vd;

            double desired_w = bal[vi][j].w;
            double current_w = amr[vo][vd];
            double ratio = current_w/desired_w;

            ratio_sum = ratio_sum + ratio;
    }

    return abs(ratio_sum - 1.0);
}


double calculate_score(vector<vector<double>> &amr, vector<vector<Edge>> &bal) {

    // Calculates the overall score given am amr.
    
    double min_ratio = numeric_limits<double>::max();
    double max_ratio = -1.0*numeric_limits<double>::max();

    // Loop over vertexes.
    for(int i = 0; i < bal.size(); i++)
        update_ratios(i, min_ratio, max_ratio, amr, bal);

    return min_ratio/max_ratio;
}


double calculate_partial_score(int &step, vector<vector<double>> &amr, vector<vector<Edge>> &bal) {

    // Calculates the overall score given am amr.
    
    double min_ratio = numeric_limits<double>::max();
    double max_ratio = -1.0*numeric_limits<double>::max();

    // Loop over vertexes.
    //fprintf(stderr, "i: ");
    for(int i = 0; i < bal.size(); i = i + step) {
        //fprintf(stderr, "%d ", i);
        update_ratios(i, min_ratio, max_ratio, amr, bal);
    }
    //fprintf(stderr, "\n");
    
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

    if (vm[nx][ny] == true) // || vm[ny][nx] == true)
        return false;

    vm[ox][oy] = false;
    //vm[oy][ox] = false;

    vm[nx][ny] = true;
    //vm[ny][nx] = true;    

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

    //fprintf(stderr, "dx %d dy %d\n", dx, dy);
    //fprintf(stderr, "v.x %d v.y %d\n", v.x, v.y);

    if ((dx == 0) && (dy == 0))
        return false;

    int nx = v.x + dx;

    //fprintf(stderr, "%d %d\n", nx, ny);

    if ((nx > BOARD_SIZE - 1) || (nx < 0))
        return false;

    int ny = v.y + dy;

    if ((ny > BOARD_SIZE - 1) || (ny < 0))
        return false;
    
    bool apv = approve_visit(v.x, v.y, nx, ny, vm);

    ////fprintf(stderr, "%d\n", apv);

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



inline double metropolis_ratio(double &ns, double &os, double &T) {

    // ns - new score
    // os - old score
    // T - temperature

    // ns - os when we maximize
    // os - ns when we minimize

    return exp( (os - ns) / T );
}



vector<Vertex> sa(vector<Vertex> &vp, 
                  vector<double> &ratio_array, 
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

    double T = 10.0; // temperature
    double tT = 0.1; // termination temperature
    double tdr = 0.7; // temperature decrease rate    
    double nI = 100000; // number of iterations per temperature step

    //double os = calculate_score(amr, bal); // old score
    //vector<double> minimal_score(N, numeric_limits<double>::max());

    auto minmax_os = minmax_element(ratio_array.begin(), ratio_array.end());
    double os = (*minmax_os.first)/(*minmax_os.second);
    double minimal_score = 0.0;

    ////fprintf(stderr, "minimal_score = %f\n", minimal_score);

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    while(T > tT) {

        for(int i = 0; i < nI; i++) {

            //fprintf(stderr, "i = %d\n", i);

            chrono::high_resolution_clock::time_point t11 = chrono::high_resolution_clock::now();

            int n_v = 1;
            vector<int> vi_vec(n_v, 0);
            vector<int> ox_vec(n_v, 0);
            vector<int> oy_vec(n_v, 0);

            for(int k = 0; k < n_v; k++) {

                vi_vec[k] = choose_vertex(g);
                ox_vec[k] = vp[vi_vec[k]].x;
                oy_vec[k] = vp[vi_vec[k]].y;
           

                while(true) {

                    ////fprintf(stderr, "Making move...\n");
                    bool made_move = random_vertex_position_update(vp[vi_vec[k]], vm, lb, rb);

                    if (made_move == true) {          
                        //update_vertex_matrix_rep(vi_vec[k], amr, bal, vp);
                        update_ratio_array(vi_vec[k], ratio_array, bal, vp);
                        break;
                    } else
                        continue;
                }
            ////fprintf(stderr, "%f\n", vs);
            }

            chrono::high_resolution_clock::time_point t22 = chrono::high_resolution_clock::now();
            int elapsed_time2211 = chrono::duration_cast<chrono::microseconds>( t22 - t11 ).count();        
            //fprintf(stderr, "Time 2211: %f s\n", (double)elapsed_time2211/1000000.0);


            chrono::high_resolution_clock::time_point t111 = chrono::high_resolution_clock::now();
            auto minmax_ns = minmax_element(ratio_array.begin(), ratio_array.end());
            double ns = (*minmax_ns.first)/(*minmax_ns.second);

            chrono::high_resolution_clock::time_point t222 = chrono::high_resolution_clock::now();
            int elapsed_time222111 = chrono::duration_cast<chrono::microseconds>( t222 - t111 ).count();        
            //fprintf(stderr, "Time 222111: %f s\n", (double)elapsed_time222111/1000000.0);

            //fprintf(stderr, "os: %f, ns: %f\n", os, ns);
            //double ns = calculate_score(amr, bal);
            double p = metropolis_ratio(ns, os, T);

            //fprintf(stderr, "new in vp; %d, %d\n", vp[vi_vec[0]].x, vp[vi_vec[0]].y );
            if ( p > uniform(g) ) {

                // Update optimal solution
                
                //fprintf(stderr, "Here\n");
                //double partial_score = calculate_partial_score(step, amr, bal);
                //double partial_score = calculate_score(amr, bal);       
                //fprintf(stderr, "There\n");       
                //fprintf(stderr, "after partial score new in vp; %d, %d\n", vp[vi_vec[0]].x, vp[vi_vec[0]].y );
                if ( minimal_score < ns ) {
                //if ( ns < os ) {
                    minimal_score = ns;
                    optimal_solution = vp;
                    //fprintf(stderr, "in if new in vp; %d, %d\n", vp[vi_vec[0]].x, vp[vi_vec[0]].y );
                }
     
            } else {
                // New state not accepted.
                // Revert back to old state.

                // Revert true/false in vm.
                for(int k = 0; k < n_v; k++) {

                    approve_visit(vp[vi_vec[k]].x, vp[vi_vec[k]].y, ox_vec[k], oy_vec[k], vm);
                    vp[vi_vec[k]].x = ox_vec[k];
                    vp[vi_vec[k]].y = oy_vec[k];
                    //update_vertex_matrix_rep(vi_vec[k], amr, bal, vp);
                    update_ratio_array(vi_vec[k], ratio_array, bal, vp);

                }

            } // outer if end


            //fprintf(stderr, "end of for loop vp; %d, %d\n", vp[vi_vec[0]].x, vp[vi_vec[0]].y );
        } // for loop end


        T = T*tdr;
        fprintf(stderr, "---->   ===   <----");
        fprintf(stderr, "T = %f\n", T);

    } // while loop end

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    int elapsed_time = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();        

    ////fprintf(stderr, "minimal_score = %f\n", minimal_score);
    fprintf(stderr, "Time: %f s\n", (double)elapsed_time/1000000.0);

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
            //vector<vector<double>> bmr(N, vector<double>(N, 0.0));

            //base_matrix_rep(bmr, edges);
            //print_matrix(bmr);


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

            int NE = (int)edges.size()/3;
            vector<double> ratio_array(NE, -1.0);

            fill_ratio_array(ratio_array, bal, vp);
            //print_vector(ratio_array);

            //auto minmax = minmax_element(ratio_array.begin(), ratio_array.end());
            //cerr << "Min: " << *(minmax.first) << endl;
            //cerr << "Max: " << *(minmax.second) << endl;


            //Adjustable matrix
            //vector<vector<double>> amr(N, vector<double>(N, 0.0));
            //adjustable_matrix_rep(amr, bal, vp);
            //print_matrix(amr);

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
            //fprintf(stderr, "-----\n"); 
            //fprintf(stderr, "Score = %4.16f\n", score);
            //fprintf(stderr, "Score = %4.16f\n", score);
            //fprintf(stderr, "Score = %4.16f\n", score);

            index = 3;
            random_vertex_move(index, amr, bal, vp, vm);
            score = calculate_score(amr, bal);
            //fprintf(stderr, "-----\n"); 
            //fprintf(stderr, "Score = %4.16f\n", score);
            //fprintf(stderr, "Score = %4.16f\n", score);
            //fprintf(stderr, "Score = %4.16f\n", score);

            int nx = 699;
            int ny = 699;
            deterministic_vertex_move(index, nx, ny, amr, bal, vp, vm); 
            score = calculate_score(amr, bal);
            //fprintf(stderr, "-----\n");            
            //fprintf(stderr, "Score = %4.16f\n", score);
            //fprintf(stderr, "Score = %4.16f\n", score);
            //fprintf(stderr, "Score = %4.16f\n", score);
            */

            int lb = -50;
            int rb = 50;
            vector<Vertex> optimal_solution = sa(vp, ratio_array, bal, vm, lb, rb);

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
