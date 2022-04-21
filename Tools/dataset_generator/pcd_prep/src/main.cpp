
#include <pcl/io/pcd_io.h>
#include <ctime>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/random_sample.h>
#include <pcl/common/centroid.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>


using namespace std;
typedef pcl::PointXYZ point;
typedef pcl::PointXYZRGBA pointcolor;


double gaussrand(double E, double V)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
     
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
             
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
         
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
         
    phase = 1 - phase;
 
    return X*V+E;
}

vector<string> split(const string& s, const string& sep)
{
        vector<string> result;
        string::size_type pos1, pos2;
        pos2 = s.find(sep);
        pos1 = 0;
        while(string::npos != pos2){
                result.push_back(s.substr(pos1, pos2-pos1));
                pos1 = pos2 +sep.size();
                pos2 = s.find(sep, pos1);
        }
        if(pos1 != s.length()) result.push_back(s.substr(pos1));
        return result;
}

int main(int argc,char **argv)
{
        pcl::PointCloud<pointcolor>::Ptr input (new pcl::PointCloud<pointcolor>);
        string path = argv[1];
        int poss= path.find_last_of('/');
        int posd= path.find_last_of('.');
        string fileparent(path.substr(0,poss+1));
        string filename(path.substr(poss+1,posd-poss-1));
        stringstream ss;
        ss<<"_";
        vector<string> sub=split(filename,ss.str());
        cout<<filename<<endl;
        point center;

        string ofilename;
        for(auto itr = sub.cbegin(); itr!= sub.cend()-3; itr++){
                
                ofilename+=*itr;
                ofilename+="_";
        }
        // cout<<ofilename<<endl;
        for(auto i = 0; i<3; i++){
                center.data[i]=atof(sub[sub.size()-3+i].c_str());
                // cout<<center.data[i]<<endl;
        }

        pcl::io::loadPCDFile(path,*input);
        
        pcl::PointCloud<point>::Ptr monopts (new pcl::PointCloud<point>);
        pcl::PointCloud<point>::Ptr output (new pcl::PointCloud<point>);
        pcl::PointCloud<point>::Ptr sampled (new pcl::PointCloud<point>);
        int M = input->points.size();
        cout<<"input size is:"<<M<<endl;
        
        for (int i = 0;i <M;i++)
        {
                point p;
                p.x = input->points[i].x;
                p.y = input->points[i].y;
                p.z = input->points[i].z; 
                monopts->points.push_back(p);
        }
        monopts->width = 1;
        monopts->height = M;

        int output_pts=atoi(argv[2]);

        while (monopts->size() < output_pts){
                for (int i = 0;i <M;i++)
                {
                        point p;
                        p.x = input->points[i].x + gaussrand(0, 0.001);
                        p.y = input->points[i].y + gaussrand(0, 0.001);
                        p.z = input->points[i].z + gaussrand(0, 0.001);
                        // cout<< input->points[i].x <<","<<p.x<<endl;
                        monopts->points.push_back(p);
                }
                cout<<"Upsampled size is:"<<monopts->size()<<endl;
        }

        pcl::RandomSample<point> rs;
        rs.setInputCloud(monopts);

        rs.setSample(output_pts);

        rs.filter(*sampled);

        Eigen::Vector4f ctr;
        pcl::compute3DCentroid(*sampled, ctr);

        cout<<ctr[0]<<","<<ctr[1]<<","<<ctr[2]<<endl;

        cout<<sampled->size()<<endl;
        for (int i = 0;i <output_pts;i++)
        {
                point p;
                p.x = sampled->points[i].x-ctr[0];
                p.y = sampled->points[i].y-ctr[1];
                p.z = sampled->points[i].z-ctr[2]; 
                output->points.push_back(p);
        }

        output->width = 1;
        output->height = output_pts;

        //pcl::compute3DCentroid(*output, ctr);
        //cout<<ctr[0]<<","<<ctr[1]<<","<<ctr[2]<<endl;


        center.data[0]-=ctr[0];
        center.data[1]-=ctr[1];
        center.data[2]-=ctr[2];

        // cout<<center.data[0]<<","<<center.data[1]<<","<<center.data[2]<<endl;

        ss.str("");
        ss.setf(ios_base::fixed);
        ss<<setprecision(2)<<center.data[0];
        ofilename = ofilename + ss.str() + "_";
        ss.str("");
        ss<<setprecision(2)<<center.data[1];
        ofilename = ofilename + ss.str() + "_";
        ss.str("");
        ss<<setprecision(2)<<center.data[2];
        ofilename = ofilename + ss.str() + ".pcd";
        cout<<ofilename<<endl;
        ss.clear();

        cout<< "Output size is "<<output->size()<<endl;
        pcl::io::savePCDFile(fileparent+ofilename,*output);

        // pcl::io::savePCDFile(fileparent+"_"+ofilename,*sampled);
        remove(path.c_str());
}