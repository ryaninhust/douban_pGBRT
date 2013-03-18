/*    Copyright 2009 10gen Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

/* simple_client_demo.cpp

   See also : http://dochub.mongodb.org/core/cppdrivertutorial

   How to build and run:

   (1) Using the mongoclient:
   g++ simple_client_demo.cpp -lmongoclient -lboost_thread-mt -lboost_filesystem -lboost_program_options
   ./a.out

   (2) using client_lib.cpp:
   g++ -I .. simple_client_demo.cpp mongo_client_lib.cpp -lboost_thread-mt -lboost_filesystem
   ./a.out
   */
//#ifndef STORE_H
//#define STORE_H
//
//#include <iostream>
//#include <stdio.h>
//#include <string.h>
//
//#include "StaticTree.h"
//#include "mongo/client/dbclient.h" // the mongo c++ driver
//
//using namespace std;
//using namespace mongo;
//using namespace bson;
//
//class Store {
//	public:
//		Store(const char* connection, const char* _collection);
//		~Store();
//		DBClientConnection c;
//		const char* collection;
//		BSONObjBuilder forestBuilder;
//
//
//		void insertNode(StaticNode*** layers, double learningrate, int level, int index, BSONObjBuilder &treeObj);
//		void insertTree(int k, int num_tree, StaticTree* tree, double learningrate);
//};
//
//Store::Store(const char* connection, const char* _collection) {
//	c.connect(connection);
//    collection = _collection;
//}
//
//Store::~Store() {
//}
//
//void Store::insertNode(StaticNode ***layers, double learningrate, int level, int index, BSONObjBuilder &treeObj)
//{
//	BSONObjBuilder bo;
//	StaticNode* node = layers[level][index];
//	bo.append("feature", node->feature);
//	bo.append("split", node->split);
//	bo.append("label", learningrate * node->label);
//	char key[30];
//	sprintf(key, "%d:%d", level, index);
//	treeObj.append(key, bo.obj());
//	if (node->feature > 0) {
//		insertNode(layers, learningrate, level+1, index*2, treeObj);
//		insertNode(layers, learningrate, level+1, index*2+1, treeObj);
//	}
//}
//
//void Store::insertTree(int k, int num_tree, StaticTree *tree, double learningrate)
//{
//	BSONObjBuilder nodesObjBuilder;
//	insertNode(tree->layers, learningrate, 0, 0, nodesObjBuilder);
//	BSONObjBuilder treeObjBuilder;
//	char key[30];
//	sprintf(key, "%d:%d", k, num_tree);
//	treeObjBuilder.append(key, nodesObjBuilder.obj());
//	c.insert(collection, treeObjBuilder.obj());
//}







/*
int main() {
	try {
		cout << "connecting to localhost..." << endl;
		DBClientConnection c;
		c.connect("localhost");
		cout << "connected ok" << endl;
		unsigned long long count = c.count("test.foo");
		cout << "count of exiting documents in collection test.foo : " << count << endl;

		//        bo o = BSON( "hello" << "world" );
		//        c.insert("test.foo", o);

		string e = c.getLastError();
		if( !e.empty() ) { 
			cout << "insert #1 failed: " << e << endl;
		}

		// make an index with a unique key constraint
		//c.ensureIndex("test.foo", BSON("hello"<<1), true);

		//c.insert("test.foo", o); // will cause a dup key error on "hello" field
		c.insert("test.foo", BSON("test"<<25));
		//cout << "we expect a dup key error here:" << endl;
		cout << "  " << c.getLastErrorDetailed().toString() << endl;
	} 
	catch(DBException& e) { 
		cout << "caught DBException " << e.toString() << endl;
		return 1;
	}

	return 0;
}
*/
#endif
