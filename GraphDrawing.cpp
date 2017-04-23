#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <limits>
#include <chrono>
#include <array>

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
        int vd; // destination
        double w; // weight
        int id;
        Edge() {}
        Edge(int td, double tw, int tid = -1) : vd(td), w(tw), id(tid) {} 
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
                fprintf(stderr, " -> %d (%f, %d); ", al[i][j].vd, al[i][j].w, al[i][j].id);
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

inline double distance(Vertex &vo, Vertex &vd) {

    // Get Euclidean distance between two vertexes.

    double d = sqrt( (vo.x - vd.x)*(vo.x - vd.x) + (vo.y - vd.y)*(vo.y - vd.y) );
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
        bal[vo].push_back(Edge(vd, w, id));
        bal[vd].push_back(Edge(vo, w, id));
        id++;
    }

}


void update_ratio_array(int &vi, vector<double> &ratio_array, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    for(int i = 0; i < bal[vi].size(); i++) {
        int vd = bal[vi][i].vd;
        double w = bal[vi][i].w;
        int id = bal[vi][i].id;

        double ratio = distance(vp[vi], vp[vd])/w;
        ratio_array[id] = ratio;
    }
}


void fill_ratio_array(vector<double> &ratio_array, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    for(int vi = 0; vi < vp.size(); vi++)
        update_ratio_array(vi, ratio_array, bal, vp);
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


void center_of_mass(int &vi, double &cmx, double &cmy, vector<Vertex> &vp, vector<vector<Edge>> &bal) {

    int N = bal[vi].size();

    if (N == 0)
        return;

    int xr0 = vp[vi].x;
    int yr0 = vp[vi].y;

    double total_mass = 0.0;
    for(int i = 0; i < N; i++) {
        int vd = bal[vi][i].vd;


        int xri = vp[vd].x - xr0;
        int yri = vp[vd].y - yr0;

        double w = bal[vi][i].w;

        double mass = distance(vp[vi], vp[vd])/w;

        cmx = cmx + mass*xri;
        cmy = cmy + mass*yri;

        total_mass = total_mass + mass;
    }

    cmx = cmx/total_mass + xr0;
    cmy = cmy/total_mass + yr0;
}



bool center_of_mass_vertex_position_update(int &vi, vector<Vertex> &vp, vector<vector<Edge>> &bal, vector<vector<bool>> &vm, double &std) {

    double cmx = 0.0;
    double cmy = 0.0;
    center_of_mass(vi, cmx, cmy, vp, bal);

    //fprintf(stderr, "cmx: %f, cmy %f\n", cmx, cmy);
    //if ((cmx < 0) || (cmy < 0))
    //    return false;

    random_device rd;
    mt19937 g(rd());
    normal_distribution<> ndx(cmx, std);
    normal_distribution<> ndy(cmy, std);
    

    int nx = (int)ndx(g);
    if ((nx > BOARD_SIZE - 1) || (nx < 0))
        return false;

    int ny = (int)ndy(g);
    if ((ny > BOARD_SIZE - 1) || (ny < 0))
        return false;
    
    bool apv = approve_visit(vp[vi].x, vp[vi].y, nx, ny, vm);

    if (apv == true) {
        vp[vi].x = nx;
        vp[vi].y = ny;
        return true;
    } else
        return false;

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

    int npm = 3; // n possible moves

    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<> move_switch(0, npm - 1);
    uniform_int_distribution<> choose_vertex(0, N - 1);
    uniform_int_distribution<> choose_grid_coordinate(0, N - 1);
    uniform_real_distribution<> uniform(0.0, 1.0);

    double T = 100.0; // temperature
    double tT = 1.0; // termination temperature
    double tdr = 0.7; // temperature decrease rate    
    double nI = 10000; // number of iterations per temperature step

    auto minmax_os = minmax_element(ratio_array.begin(), ratio_array.end());
    double os = (*minmax_os.first)/(*minmax_os.second);
    double minimal_score = 0.0;

    fprintf(stderr, "start os: %f\n", os);

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    while(T > tT) {

        for(int i = 0; i < nI; ++i) {

            //chrono::high_resolution_clock::time_point t11 = chrono::high_resolution_clock::now();

            int n_v = 2;
            double std = 100.0;
            vector<int> vi_vec(n_v, 0);
            vector<int> ox_vec(n_v, 0);
            vector<int> oy_vec(n_v, 0);

            for(int k = 0; k < n_v; ++k) {

                vi_vec[k] = choose_vertex(g);
                ox_vec[k] = vp[vi_vec[k]].x;
                oy_vec[k] = vp[vi_vec[k]].y;
           
                while(true) {
                    bool made_move = false;
                    int switch_flag = move_switch(g);

                    if (switch_flag == 0)
                        made_move = random_vertex_position_update(vp[vi_vec[k]], vm, lb, rb);
                    else if (switch_flag == 1)
                        made_move = center_of_mass_vertex_position_update(vi_vec[k], vp, bal, vm, std);
                    else
                        made_move = random_vertex_teleport(vp[vi_vec[k]], vm);

                    //cerr << "made_move: " << made_move << endl;
                    if (made_move == true) {          
                        update_ratio_array(vi_vec[k], ratio_array, bal, vp);
                        break;
                    } else
                        continue;
                }
            }

            //chrono::high_resolution_clock::time_point t22 = chrono::high_resolution_clock::now();
            //int elapsed_time2211 = chrono::duration_cast<chrono::microseconds>( t22 - t11 ).count();        
            //fprintf(stderr, "Time 2211: %f s\n", (double)elapsed_time2211/1000000.0);


            //chrono::high_resolution_clock::time_point t111 = chrono::high_resolution_clock::now();

            auto minmax_ns = minmax_element(ratio_array.begin(), ratio_array.end());
            double ns = (*minmax_ns.first)/(*minmax_ns.second);

            //chrono::high_resolution_clock::time_point t222 = chrono::high_resolution_clock::now();
            //int elapsed_time222111 = chrono::duration_cast<chrono::microseconds>( t222 - t111 ).count();        
            //fprintf(stderr, "Time 222111: %f s\n", (double)elapsed_time222111/1000000.0);

            double p = metropolis_ratio(ns, os, T);

            if ( p > uniform(g) ) {

                // Update optimal solution
                if ( minimal_score < ns ) {
                    minimal_score = ns;
                    optimal_solution = vp;
                }
     
            } else {
                // New state not accepted.
                // Revert back to old state.
                for(int k = 0; k < n_v; ++k) {

                    approve_visit(vp[vi_vec[k]].x, vp[vi_vec[k]].y, ox_vec[k], oy_vec[k], vm);
                    vp[vi_vec[k]].x = ox_vec[k];
                    vp[vi_vec[k]].y = oy_vec[k];
                    //update_vertex_matrix_rep(vi_vec[k], amr, bal, vp);
                    update_ratio_array(vi_vec[k], ratio_array, bal, vp);

                }

            } // outer if end

        } // for loop end


        T = T*tdr;
        //fprintf(stderr, "---->   ===   <----");
        //fprintf(stderr, "T = %f\n", T);
        //fprintf(stderr, "minimal_score = %f\n", minimal_score);

    } // while loop end

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    int elapsed_time = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();        

    fprintf(stderr, "minimal_score = %f\n", minimal_score);
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

            int vi = 0;
            double cmx = 0;
            double cmy = 0;
            center_of_mass(vi, cmx, cmy, vp, bal);
            fprintf(stderr, "f: %f; s: %f\n", cmx, cmy);


            fill_ratio_array(ratio_array, bal, vp);
            //print_vector(ratio_array);

            int lb = -100;
            int rb = 100;
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
