/*
 * threaded_print.h
 *
 *  Created on: Jan 11, 2020
 *      Author: panpan
 */

#ifndef THREADED_PRINT_H_
#define THREADED_PRINT_H_

#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include "debug_util.h"
#include "despot/util/logging.h"
#include "despot/GPUcore/thread_globals.h"

#define terr ThreadStream(std::cerr)
#define tout ThreadStream(std::cout)

/**
 * Thread-safe std::ostream class.
 *
 * Usage:
 *    tout << "Hello world!" << std::endl;
 *    terr << "Hello world!" << std::endl;
 */

class ThreadStream : public std::ostringstream
{
    public:
        ThreadStream(std::ostream& os) : os_(os)
        {
//        	if (Globals::MapThread(this_thread::get_id())==0) {
				if (logging::level() >= thresh) {
					imbue(os.getloc());
					precision(os.precision());
					width(os.width());
					setf(std::ios::fixed, std::ios::floatfield);
				}
//        	}
        }

        ~ThreadStream() {
//        	if (Globals::MapThread(this_thread::get_id())==0) {
				if (logging::level() >= thresh) {
					std::lock_guard<std::mutex> guard(_mutex_threadstream);

					os_ << "~~~~~~~~~~~~~~~~~~~~~~~~~~ <" << Globals::MapThread(this_thread::get_id()) << "> "
							<< this->str();
	//				os_ << "<" << this_thread::get_id() << "> "
	//										<< this->str();
				}
//        	}
        }

    private:
        static std::mutex _mutex_threadstream;
        std::ostream& os_;
        const int thresh = logging::INFO;
};

#define TDEBUG(msg) { std::string str = msg; \
				tout << string_sprintf("%s, in %s, at file %s_line_%d \n", str.c_str(), __FUNCTION__, __FILE__, __LINE__); }


#endif /* THREADED_PRINT_H_ */
