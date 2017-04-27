#include <cstdio>
#include <set>
#include <map>
#include <algorithm>
#include <random>
#include <iostream>

const double EPSILON = 1e-4;

using namespace std;


class Ratio {
    public:
        int id;
        double r;
        Ratio(): id(-1), r(-1.0) {}
        Ratio(int ti, double tr): id(ti), r(tr) {}

        friend bool operator< (const Ratio &r1, const Ratio &r2);
};  


bool operator< (const Ratio &left, const Ratio &right)
{
    return left.r < right.r;
}


inline bool double_equal(const double &d1, const double &d2) {

    if( abs(d1 - d2) < EPSILON )
        return true;
    else
        return false;

}


class RatioFinder {
    public:
        double r;
        RatioFinder(const double &tr) : r(tr) {}

        bool operator()(const Ratio &ratio) {
            return double_equal(ratio.r, r);
        }

};


void populate_map_and_multiset(map<int, double> &m, multiset<Ratio> &ms, int &N) {

    random_device rd;
    //mt19937 g(rd());
    mt19937 g(1);    


    uniform_int_distribution<> coin_flip(0, 1);
    uniform_real_distribution<> uniform(0.0, 1.0);

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if(i <= j)
                continue;

            if (coin_flip(g) == 1) {
                int id = i*N+j;
                double r = uniform(g);

                m[id] = r;
                ms.insert(Ratio(id, r));
                
            }

        }
    }

}


void update_ratio(double &n_r, double &o_r, map<int, double> &m, multiset<Ratio> &ms) {

    if (double_equal(n_r, o_r) == true)
        return;

    multiset<Ratio>::iterator msit;
    
    while (true) {

        msit = find_if( ms.begin(), ms.end(), RatioFinder(o_r) );

        if(msit != ms.end()) {
            int id = msit->id;

            ms.erase(msit);         
            ms.insert(Ratio(id, n_r));

            m[id] = n_r;

        } else
            break;

    }

}


void print_multiset(multiset<Ratio> &ms) {

    multiset<Ratio>::iterator msit;

    for(msit = ms.begin(); msit != ms.end(); msit++) {
        printf("id: %d, r: %f\n", msit->id, msit->r);
    }

}


void print_vecs(vector<Ratio> &vec, vector<Ratio*> &pvec) {

    printf("Vec\n");
    for(int i = 0; i < vec.size(); i++)
        cout << vec[i].id << " -- " << vec[i].r << endl;

    printf("pVec\n");
    for(int i = 0; i < vec.size(); i++)
        cout << pvec[i]->id << " -- " << pvec[i]->r << endl;
}


int main() {

    multiset<Ratio> ts {Ratio(0, 0.5), Ratio(1, 0.7), Ratio(2, 0.55), Ratio(3, 0.2), Ratio(4, 0.5)};

    printf("min %d %f\n", (ts.begin())->id, (ts.begin())->r );
    printf("max %d %f\n", (ts.rbegin())->id, (ts.rbegin())->r );
    multiset<Ratio>::iterator rit = find_if( ts.begin(), ts.end(), RatioFinder(0.5) );
    if(rit != ts.end())
        printf("%d %f\n", rit->id, rit->r);
    else
        printf("Not found!");


    print_multiset(ts);

    map<int, double> m;
    multiset<Ratio> ms;
    printf("size of before population m: %d\n", (int)m.size());

    int N = 6;
    populate_map_and_multiset(m, ms, N);
    printf("size of after  population m: %d\n", (int)m.size());

    print_multiset(ms);

    double n_r = -1.0;
    double o_r = 0.419195;
    update_ratio(n_r, o_r, m, ms);

    printf("Updating ratio...\n");
    print_multiset(ms);


    printf("---> Pointer check <---\n");
    vector<Ratio> vec {Ratio(1, 0.4), Ratio(2, 0.5), Ratio(3, 0.3), Ratio(4, 0.45), Ratio(5, 0.1)};
    vector<Ratio*> pvec(vec.size());

    for(int i = 0; i < vec.size(); i++)
        pvec[i] = &vec[i];

    printf("Before sorting...");
    print_vecs(vec, pvec);

    printf("--> Sorting <--\n");
    auto cf = [](const Ratio *r1, const Ratio *r2){ return r1->r < r2->r; };
    sort(pvec.begin(), pvec.end(), cf);

    print_vecs(vec, pvec);

    printf("Changing one element...");
    vec[2].r = 0.475;

    print_vecs(vec, pvec);

    printf("--> Sorting one more time <--\n");
    sort(pvec.begin(), pvec.end(), cf);

    print_vecs(vec, pvec);

}









