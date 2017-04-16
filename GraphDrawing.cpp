#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <limits>

const int BOARD_SIZE = 700;
const bool VERBOSE_MODE = true;

using namespace std;


class Vertex {
    public:
        int x;
        int y;
        Vertex() {}
        Vertex(int tx, int ty): x(tx), y(ty) {}
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
    return d;
}


void base_adjacency_list_rep(vector<vector<Edge>> &bal, int &N, vector<int> &edges) {

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


void refresh_vertex_adjustable_adjacency_list(int &vi, vector<vector<Edge>> &aal, vector<Vertex> &vp) {

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


void refresh_adjustable_adjacency_list(vector<int> vua, vector<vector<Edge>> &aal, vector<Vertex> &vp) {

    // This function assumes that the state of vp has changed.
    // The positions of vertexes given in the vertex update array (vua)
    // have been changed.
    // The corresponding weights in aal have to be updated.
    // This function performs the required update.

    for(int i = 0; i < vua.size(); i++) 
        refresh_vertex_adjustable_adjacency_list(vua[i], aal, vp);

}


void refresh_vertex_matrix_rep(int &vi, vector<vector<double>> &amr, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    for(int i = 0; i < bal[vi].size(); i++) {
        int vo = bal[vi][i].vo;
        int vd = bal[vi][i].vd;

        double d = distance(vp[vo], vp[vd]);
        amr[vo][vd] = d;
        amr[vd][vo] = d;
    }
}


void refresh_matrix_rep(vector<int> vua, vector<vector<double>> &amr, vector<vector<Edge>> &bal, vector<Vertex> &vp) {

    for(int i = 0; i < vua.size(); i++) 
        refresh_vertex_matrix_rep(vua[i], amr, bal, vp);

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
            if (max_ratio < ratio )
                max_ratio = ratio;
    }
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
            base_adjacency_list_rep(bal, N, edges);
            print_adjacency_list(bal);

            //Adjustable matrix
            vector<vector<double>> amr(N, vector<double>(N, 0.0));
            adjustable_matrix_rep(amr, bal, vp);
            print_matrix(amr);

            //Adjustable adjacency list
            vector<vector<Edge>> aal(N, vector<Edge>(0));
            adjustable_adjacency_list_rep(aal, bal, vp);
            print_adjacency_list(aal);

            // Changeing the position of vertex 3.
            int index = 4;
            vp[index].x = 100;
            vp[index].y = 200;
            //refresh_vertex_adjustable_adjacency_list(index, aal, vp);
            //print_adjacency_list(aal);
            refresh_vertex_matrix_rep(index, amr, bal, vp);
            print_matrix(amr);


            vector<int> vua = {0, 1};
            vp[vua[0]].x = 300;
            vp[vua[0]].y = 300;
            vp[vua[1]].x = 400;
            vp[vua[1]].y = 400;
            //refresh_adjustable_adjacency_list(vua, aal, vp);
            //print_adjacency_list(aal);
            refresh_matrix_rep(vua, amr, bal, vp);
            print_matrix(amr);

            double score = calculate_score(amr, bal);
            cerr << "Score: " << score << endl;
            cerr << "Score: " << score << endl;
            cerr << "Score: " << score << endl;

            vector<int> ret(2*N);
            dispatch_vertex_positions(ret, vp);
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
