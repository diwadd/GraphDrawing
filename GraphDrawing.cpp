#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include <random>

const int BOARD_SIZE = 700;

using namespace std;


class Edge {
    public:
        int vo;
        int vd;
        double w;
        Edge() {}
        Edge(int o, int d, double w) : vo(o), vd(d), w(w) {} 
};


void print_adjacency_list(vector<vector<Edge>> &al) {

    fprintf(stderr, "Printing adjacency list...\n");
    for(int i = 0; i < al.size(); i++) {
        fprintf(stderr, "Vertex %d: ", i);
        for(int j = 0; j < al[i].size(); j++) {
            fprintf(stderr, "%d -> %d (%f); ", al[i][j].vo, al[i][j].vd, al[i][j].w);
        }
        fprintf(stderr, "\n");
    }

}


void print_matrix(vector<vector<double>> &mr) {

    fprintf(stderr, "Printing matrix...\n");
    for(int i = 0; i < mr.size(); i++) {
        for(int j = 0; j < mr[i].size(); j++) {
            fprintf(stderr, "%4.2f ", mr[i][j]);
        }
        fprintf(stderr, "\n");
    }
}

double distance(pair<int, int> &vo, pair<int, int> &vd) {

    int vox = vo.first;
    int voy = vo.second;

    int vdx = vd.first;
    int vdy = vd.second;

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


void adjustable_adjacency_list_rep(vector<vector<Edge>> &aal, vector<vector<Edge>> &bal, vector<pair<int, int>> &vp) {

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

void adjustable_matrix_repo(vector<vector<double>> &amr, vector<vector<Edge>> &bal, vector<pair<int, int>> &vp) {

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


void initialize_vertex_positions(vector<pair<int, int>> &vp, vector<vector<bool>> &vm) {

    // Assign random positions to the vertexes.
    // Mark the used positions in vm.

    random_device rd;
    mt19937 g(rd());
    uniform_int_distribution<> uid(0, BOARD_SIZE - 1);

    for(int i = 0; i < vp.size(); i++) {
        while(true) {        
            int f = uid(g);
            int s = uid(g); 

            if ((vm[f][s] == true) || (vm[s][f] == true))               
                continue;

            vm[f][s] = true;
            vm[s][f] = true;

            vp[i].first = f;
            vp[i].second = s;
            break;

        } // while end

    } // for end

}


void dispatch_vertex_positions(vector<int> &ret, vector<pair<int, int>> &vp) {

    // Converts the custom vertex positions representation into
    // the one required by the competition i. e. vector of ints.

    for(int i = 0; i < vp.size(); i++) {
        ret[2*i + 0] = vp[i].first;
        ret[2*i + 1] = vp[i].second;
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
            vector<pair<int, int>> vp(N);

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
            adjustable_matrix_repo(amr, bal, vp);
            print_matrix(amr);

            //Adjustable adjacency list
            vector<vector<Edge>> aal(N, vector<Edge>(0));
            adjustable_adjacency_list_rep(aal, bal, vp);
            print_adjacency_list(aal);

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
