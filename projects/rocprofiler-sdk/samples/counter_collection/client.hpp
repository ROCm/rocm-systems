#pragma once

#include <iostream>

#define CLIENT_API __attribute__((visibility("default")))

int
start() CLIENT_API;