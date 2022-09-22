#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <msg_builder/car_info.h>
#include <msg_builder/ped_info.h>
#include <msg_builder/peds_info.h>
#include <msg_builder/peds_car_info.h>
#include <iostream>
#include <vector>
#include <RVO.h>
#include "coord.h"

struct Car {
    Car(){
        vel = 1.0;
    }
    COORD pos;
    double yaw;
    double vel;
};

struct Ped {
    Ped(){
        vel = 1.2;
    }
    Ped(COORD a, int b, int c) {
        pos = a;
        goal = b;
        id = c;
        vel = 1.2;
    }
    COORD pos; //pos
    int goal;  //goal
    int id;   //id
    double vel;
};

class PedsSystem {
public:
    Car car;
    std::vector<Ped> peds;

    RVO::RVOSimulator* ped_sim_;

    ros::Subscriber peds_car_info_sub;
    ros::Subscriber action_sub;
    ros::Publisher peds_info_pub;
    std::vector<COORD> goals;
    bool initialized;

    float freq;

    PedsSystem(){
        initialized = false;
        freq = 3.0;
        goals = { // Unity airport departure
            COORD(-197.80, -134.80), // phora
            COORD(-180.15, -137.54), // Nana?
            COORD(-169.33, -141.1), // gate 1,2,3 
            COORD(-174.8, -148.53), // Cafe2
            COORD(-201.55, -148.53), //Cafe1
            COORD(-216.57, -145), // Gate 4,5,6
            COORD(-1, -1) // stop
        };

        ped_sim_ = new RVO::RVOSimulator();
    
        // Specify global time step of the simulation.
        ped_sim_->setTimeStep(1.0f/freq);    

        // Specify default parameters for agents that are subsequently added.
        //ped_sim_->setAgentDefaults(5.0f, 8, 10.0f, 5.0f, 0.5f, 2.0f);
        ped_sim_->setAgentDefaults(1.5f, 1, 3.0f, 6.0f, 0.12f, 3.0f);

        addObstacle();
    }

    void spin() {
        ros::NodeHandle nh;
        peds_car_info_sub = nh.subscribe("peds_car_info", 1, &PedsSystem::pedsCarCallBack, this);
        action_sub = nh.subscribe("cmd_vel_pomdp", 1, &PedsSystem::actionCallBack, this);
        peds_info_pub = nh.advertise<msg_builder::peds_info>("peds_info",1);
        ros::spin();
    }

    int findPedwithID(int id){
        int i=0;
        for(i; i<peds.size(); i++){
            if(id == peds[i].id) return i;
        }
        return i;
    }

    void pedsCarCallBack(msg_builder::peds_car_infoConstPtr peds_car_ptr) {

        car.pos.x = peds_car_ptr->car.car_pos.x;
        car.pos.y = peds_car_ptr->car.car_pos.y;
        car.yaw = peds_car_ptr->car.car_yaw;

        Ped tmp_ped;
        for (int i = 0; i < peds_car_ptr->peds.size(); i++)
        {
            tmp_ped.pos.x = peds_car_ptr->peds[i].ped_pos.x;
            tmp_ped.pos.y = peds_car_ptr->peds[i].ped_pos.y;
            tmp_ped.goal = peds_car_ptr->peds[i].ped_goal_id;
            tmp_ped.id = peds_car_ptr->peds[i].ped_id;
            //tmp_ped.vel = peds_car_ptr->peds[i].ped_speed;

            int index = findPedwithID(tmp_ped.id);
            if(index >= peds.size()) peds.push_back(tmp_ped);
            else peds[index] = tmp_ped;
        }

        initialized = true;
    }

    void actionCallBack(geometry_msgs::TwistConstPtr action) {
        if (initialized == false) return;
        if(action->linear.x==-1) car.vel = 0;
        car.vel = action->linear.x;
        
        RVO2PedStep();

        msg_builder::peds_info peds_info_msg;
        getPedsInfoMsg(peds_info_msg);
        peds_info_pub.publish(peds_info_msg);
        peds.clear();
    }

    void getPedsInfoMsg(msg_builder::peds_info & peds_info_msg){

        msg_builder::ped_info tmp_single_ped;

        for(int i=0; i< peds.size(); i++){
            tmp_single_ped.ped_pos.x = peds[i].pos.x;
            tmp_single_ped.ped_pos.y = peds[i].pos.y;
            tmp_single_ped.ped_pos.z = 0;
            tmp_single_ped.ped_goal_id = peds[i].goal;
            tmp_single_ped.ped_id = peds[i].id;
            tmp_single_ped.ped_speed = peds[i].vel;
            peds_info_msg.peds.push_back(tmp_single_ped);
        }

    }

    void RVO2PedStep(){

        ped_sim_->clearAllAgents();

        //adding pedestrians
        for(int i=0; i<peds.size(); i++){
            ped_sim_->addAgent(RVO::Vector2(peds[i].pos.x, peds[i].pos.y));
        }

        //addAgent (const Vector2 &position, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed)
        ped_sim_->addAgent(RVO::Vector2(car.pos.x, car.pos.y), 3.0f, 1, 3.0f, 5.0f, 0.8f, 3.0f, RVO::Vector2(), "vehicle");
        ped_sim_->setAgentPrefVelocity(peds.size(), RVO::Vector2(car.vel * cos(car.yaw), car.vel * sin(car.yaw))); // the num_ped-th pedestrian is the car. set its prefered velocity

        // Set the preferred velocity for each agent.
        for (size_t i = 0; i < peds.size(); i++) {
            int goal_id = peds[i].goal;
            if (goal_id >= goals.size()-1) { /// stop intention
                ped_sim_->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
            }
            else{
                RVO::Vector2 goal(goals[goal_id].x, goals[goal_id].y);
                if ( absSq(goal - ped_sim_->getAgentPosition(i)) < ped_sim_->getAgentRadius(i) * ped_sim_->getAgentRadius(i) ) {
                    // Agent is within one radius of its goal, set preferred velocity to zero
                    //ped_sim_->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
                    ped_sim_->setAgentPrefVelocity(i, normalize(goal - ped_sim_->getAgentPosition(i)));
                } else {
                    // Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
                    ped_sim_->setAgentPrefVelocity(i, normalize(goal - ped_sim_->getAgentPosition(i))*1.5);
                }
            }
            
        }

        ped_sim_->doStep();

        for(int i=0; i<peds.size(); i++){
            peds[i].vel = freq*sqrt((ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)*(ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)
                +(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y)*(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y));
            if (peds[i].vel < 0.2 && peds[i].vel > 0){
                peds[i].pos.x=ped_sim_->getAgentPosition(i).x() + (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)*(1/peds[i].vel); //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
                peds[i].pos.y=ped_sim_->getAgentPosition(i).y() + (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)*(1/peds[i].vel);//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
                peds[i].vel = freq*sqrt((ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)*(ped_sim_->getAgentPosition(i).x()-peds[i].pos.x)
                    +(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y)*(ped_sim_->getAgentPosition(i).y()-peds[i].pos.y));
            }
            // if(peds[i].pos.x==ped_sim_->getAgentPosition(i).x()){
            //     std::cout<<"ped not changed!!!!!!"<<std::endl;
            // }
            // else{
            //     std::cout<<"changed$$$$$$"<<std::endl;
            // }
            //std::cout<<peds[i].id<<"  "<<peds[i].pos.x<<"  "<<ped_sim_->getAgentPosition(i).x();
            // if(peds[i].id == 0){
            //     std::cout<<peds[i].id<<"  "<<peds[i].pos.x<<"  "<<ped_sim_->getAgentPosition(i).x()<<"  "<<peds[i].vel<<std::endl;
            //     std::cout<<"  "<<peds[i].pos.y<<"  "<<ped_sim_->getAgentPosition(i).y()<<"  "<<peds[i].goal<<std::endl<<std::endl;

            // }
            else{
                peds[i].pos.x=ped_sim_->getAgentPosition(i).x();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).x() - peds[i].pos.x)/5.0; //random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
                peds[i].pos.y=ped_sim_->getAgentPosition(i).y();// + random.NextGaussian() * (ped_sim_->getAgentPosition(i).y() - peds[i].pos.y)/5.0;//random.NextGaussian() * ModelParams::NOISE_PED_POS / freq;
            } 
        }
    }


    void addObstacle(){
        std::vector<RVO::Vector2> obstacle[12];

        obstacle[0].push_back(RVO::Vector2(-222.55,-137.84));
        obstacle[0].push_back(RVO::Vector2(-203.23,-138.35));
        obstacle[0].push_back(RVO::Vector2(-202.49,-127));
        obstacle[0].push_back(RVO::Vector2(-222.33,-127));

        obstacle[1].push_back(RVO::Vector2(-194.3,-137.87));
        obstacle[1].push_back(RVO::Vector2(-181.8,-138));
        obstacle[1].push_back(RVO::Vector2(-181.5,-127));
        obstacle[1].push_back(RVO::Vector2(-194.3,-127));

        obstacle[2].push_back(RVO::Vector2(-178.5,-137.66));
        obstacle[2].push_back(RVO::Vector2(-164.95,-137.66));
        obstacle[2].push_back(RVO::Vector2(-164.95,-127));
        obstacle[2].push_back(RVO::Vector2(-178.5,-127));

        obstacle[3].push_back(RVO::Vector2(-166.65,-148.05));
        obstacle[3].push_back(RVO::Vector2(-164,-148.05));
        obstacle[3].push_back(RVO::Vector2(-164,-138));
        obstacle[3].push_back(RVO::Vector2(-166.65,-138));

        obstacle[4].push_back(RVO::Vector2(-172.06,-156));
        obstacle[4].push_back(RVO::Vector2(-166,-156));
        obstacle[4].push_back(RVO::Vector2(-166,-148.25));
        obstacle[4].push_back(RVO::Vector2(-172.06,-148.25));

        obstacle[5].push_back(RVO::Vector2(-197.13,-156));
        obstacle[5].push_back(RVO::Vector2(-181.14,-156));
        obstacle[5].push_back(RVO::Vector2(-181.14,-148.65));
        obstacle[5].push_back(RVO::Vector2(-197.13,-148.65));

        obstacle[6].push_back(RVO::Vector2(-222.33,-156));
        obstacle[6].push_back(RVO::Vector2(-204.66,-156));
        obstacle[6].push_back(RVO::Vector2(-204.66,-148.28));
        obstacle[6].push_back(RVO::Vector2(-222.33,-148.28));

        obstacle[7].push_back(RVO::Vector2(-214.4,-143.25));
        obstacle[7].push_back(RVO::Vector2(-213.5,-143.25));
        obstacle[7].push_back(RVO::Vector2(-213.5,-142.4));
        obstacle[7].push_back(RVO::Vector2(-214.4,-142.4));

        obstacle[8].push_back(RVO::Vector2(-209.66,-144.35));
        obstacle[8].push_back(RVO::Vector2(-208.11,-144.35));
        obstacle[8].push_back(RVO::Vector2(-208.11,-142.8));
        obstacle[8].push_back(RVO::Vector2(-209.66,-142.8));

        obstacle[9].push_back(RVO::Vector2(-198.58,-144.2));
        obstacle[9].push_back(RVO::Vector2(-197.2,-144.2));
        obstacle[9].push_back(RVO::Vector2(-197.2,-142.92));
        obstacle[9].push_back(RVO::Vector2(-198.58,-142.92));

        obstacle[10].push_back(RVO::Vector2(-184.19,-143.88));
        obstacle[10].push_back(RVO::Vector2(-183.01,-143.87));
        obstacle[10].push_back(RVO::Vector2(-181.5,-141.9));
        obstacle[10].push_back(RVO::Vector2(-184.19,-142.53));

        obstacle[11].push_back(RVO::Vector2(-176,-143.69));
        obstacle[11].push_back(RVO::Vector2(-174.43,-143.69));
        obstacle[11].push_back(RVO::Vector2(-174.43,-142));
        obstacle[11].push_back(RVO::Vector2(-176,-142));


        for (int i=0; i<12; i++){
           ped_sim_->addObstacle(obstacle[i]);
        }

        /* Process the obstacles so that they are accounted for in the simulation. */
        ped_sim_->processObstacles();
    }

};

int main(int argc,char**argv)
{
	ros::init(argc,argv,"gamma");
    PedsSystem peds_system;
    peds_system.spin();
	return 0;
}
