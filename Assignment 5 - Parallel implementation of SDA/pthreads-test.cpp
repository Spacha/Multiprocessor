#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <inttypes.h>

using namespace std;

#define NUM_THREADS 8

void *PrintHello(void *threadid) {
   intptr_t tid;
   tid = (intptr_t)threadid;
   cout << "Hello World! Thread ID, " << tid << endl;
   pthread_exit(NULL);

   return nullptr;
}

int main () {
   pthread_t threads[NUM_THREADS];
   int rc;
   intptr_t i;
   
   for( i = 0; i < NUM_THREADS; i++ )
   {
      cout << "main() : creating thread, " << i << endl;
      rc = pthread_create(&threads[i], NULL, PrintHello, (void *)i);
      
      if (rc) {
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
      }
   }

   for( i = 0; i < NUM_THREADS; i++ )
   {
      pthread_join(threads[i], NULL);
      cout << "Thread " << i << " joined." << endl;
   }

   cout << "All done!" << endl;

   pthread_exit(NULL);
}
