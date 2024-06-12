#include "robot.h"
#include "ultis.h"
// All the webots classes are defined in the "webots" namespace
int main(int argc, char **argv) {
  TEBControl *controller = new TEBControl();
  controller->run();
  delete controller;
  return 0;
}
