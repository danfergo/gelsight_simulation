#include <iostream>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <ros/callback_queue.h>

#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/joint_state_interface.h>
#include <controller_manager/controller_manager.h>
#include <hardware_interface/robot_hw.h>

#include <position_controllers/joint_group_position_controller.h>

#include <serial/serial.h>

#include <memory>
#include <string>
#include <stdexcept>

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

class FDMPrinter : public hardware_interface::RobotHW
{
private:
  serial::Serial * port;
  hardware_interface::JointStateInterface jnt_state_interface;
  hardware_interface::PositionJointInterface jnt_pos_interface;

  double cmd[3];
  double pos[3];
  double vel[3];
  double eff[3];

  ros::Duration elapsed_time_;
  boost::shared_ptr<controller_manager::ControllerManager> controller_manager_;
  ros::Timer non_realtime_loop_;

public:
  FDMPrinter(ros::NodeHandle &nh_)
 {
   serial::Timeout t = serial::Timeout::simpleTimeout(250);
   port = new serial::Serial("/dev/ttyUSB0", 115200);


    cmd[0] = 0;
    cmd[1] = 0;
    cmd[2] = 0;


   // connect and register the joint state interface
   hardware_interface::JointStateHandle state_handle_x("x_axis", &pos[0], &vel[0], &eff[0]);
   jnt_state_interface.registerHandle(state_handle_x);

   hardware_interface::JointStateHandle state_handle_y("y_axis", &pos[1], &vel[1], &eff[1]);
   jnt_state_interface.registerHandle(state_handle_y);

   hardware_interface::JointStateHandle state_handle_z("z_axis", &pos[2], &vel[2], &eff[2]);
   jnt_state_interface.registerHandle(state_handle_z);

   registerInterface(&jnt_state_interface);

   // connect and register the joint position interface
   hardware_interface::JointHandle pos_handle_x(jnt_state_interface.getHandle("x_axis"), &cmd[0]);
   jnt_pos_interface.registerHandle(pos_handle_x);

   hardware_interface::JointHandle pos_handle_y(jnt_state_interface.getHandle("y_axis"), &cmd[1]);
   jnt_pos_interface.registerHandle(pos_handle_y);

   hardware_interface::JointHandle pos_handle_z(jnt_state_interface.getHandle("z_axis"), &cmd[2]);
   jnt_pos_interface.registerHandle(pos_handle_z);

   registerInterface(&jnt_pos_interface);



   controller_manager_.reset(new controller_manager::ControllerManager(this, nh_));
   ros::Duration update_freq = ros::Duration(1.0/30);
   non_realtime_loop_ = nh_.createTimer(update_freq, &FDMPrinter::update, this);


   ros::Duration period = ros::Duration(5);
   period.sleep();
   this->goHome();
   period.sleep();
  }


   void goHome() {
       std:: cout << "[write] " << "G28 \n";
       port->write("G28 \n");
       port->flush();
       std::cout << "flushed" << std::endl;

//      ros::Duration period = ros::Duration(5);
//      period.sleep();
//      period.sleep();

   }


   void read(){
        pos[0] = cmd[0];
        pos[1] = cmd[1];
        pos[2] = cmd[2];
   }

   void write(ros::Duration elapsed_time){
       if(cmd[0] < 0 || cmd[1] < 0 || cmd[2] < 0){
            return;
       }

       if (cmd[0] != pos[0] || cmd[1] != pos[1] || cmd[2] != pos[2]){
           std::stringstream ss;
           ss << string_format("G0 X%.2f Y%.2f Z%.2f \n", cmd[0]*1000, cmd[1]*1000, cmd[2]*1000) << std::endl;
           std:: cout << "[write] " << ss.str();
           port->write(ss.str());
           port->flush();
           std::cout << "flushed" << std::endl;
       }
   }

   void sendGCodeCmd(const std_msgs::String::ConstPtr& msg){
       std::cout << "Send: " << msg->data << std::endl;
       port->write(msg->data + "\n");
       port->flush();
   }


   void update(const ros::TimerEvent& e) {
        elapsed_time_ = ros::Duration(e.current_real - e.last_real);
        read();
        controller_manager_->update(ros::Time::now(), elapsed_time_);
        write(elapsed_time_);
   }

   ros::Time get_time() {
        return ros::Time::now();
   }

    ros::Duration get_period() {
        return ros::Duration(1/(double)2);
    }


 ~FDMPrinter() {
    port->close();
    delete port;
 }
};


int main(int argc, char** argv) {

  ros::init(argc, argv, "fdm_printer_hardware_interface");
  ros::CallbackQueue ros_queue;


  ros::NodeHandle nh;
  nh.setCallbackQueue(&ros_queue);
  FDMPrinter printer(nh);

  // ros::Subscriber sub = nh.subscribe("gcode_script", 10, &FDMPrinter::sendGCodeCmd, &printer);
  // std::cout << "subscribed" << sub << std::endl;

  ros::MultiThreadedSpinner spinner(0);
  spinner.spin(&ros_queue);
  return 0;

  //controller_manager::ControllerManager cm(&printer);



  //ros::Duration period = printer.get_period();



  //period.sleep();
  //period.sleep();
  //period.sleep();
  //period.sleep();

  //printer.goHome();
  //period.sleep();

  //std::cout << "end" << std::endl;

  //while (ros::ok())
  //{
  //   printer.read();
  //   cm.update(printer.get_time(), period);
  //   printer.write();
  //   period.sleep();
  //   ros::spinOnce();
  //}
  //return 0;
}