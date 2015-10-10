#include "common.h"

static int numThreads = 1;

int getNumThreads() {
   return numThreads;
}

void setNumThreads(int t) {
   if (t > 0)
      numThreads = t;
}

inline uint32_t getColor(uint32_t it) {
   if (it == MAX) {
      return 0;
   }

   uint32_t R = ((it>>2)&1) | ((it>>4)&2) | ((it>>6)&4) | ((it>>8)&8) | ((it>>10)&16);
   uint32_t G = ((it>>1)&1) | ((it>>3)&2) | ((it>>5)&4) | ((it>>7)&8) | ((it>>9) &16);
   uint32_t B = ( it    &1) | ((it>>2)&2) | ((it>>4)&4) | ((it>>6)&8) | ((it>>8) &16) | ((it>>10)&32);
   return (B<<2) | (G<<11) | (R<<19);
}

int findCurrentRun() {
   struct stat st;

   int run = 0;
   if (stat("images", &st) == -1) {
      printf("images folder missing, creating it now\n");
      mkdir("images", 0755);
   } else {
      printf("looking for previous runs...\n");

      DIR *dir = opendir("images");

      struct dirent *entry = readdir(dir);
      while (entry) {
         if (entry->d_type == DT_DIR)
            if (strncmp("run_", entry->d_name, 4) == 0)
               run++;

         entry = readdir(dir);
      }

      closedir(dir);
   }

   char path[16];
   sprintf(path, "images/run_%04d", run);

   printf("creating folder %s for images\n", path);
   mkdir(path, 0755);
   return run;
}

void saveImage(int run, int len, int num, uint32_t *iters) {
   char bfType1=0x42;
   char bfType2=0x4D;
   char bfSize[4]={0x00,0x00,0x00,0x00}; 
   char bfReserved1[2]={0x00,0x00};
   char bfReserved2[2]={0x00,0x00};
   char bfOffset[4]={0x36,0x00,0x00,0x00};
   char biSize[4]={0x28,0x00,0x00,0x00};
   
   char biWidth[4] = {char (WIDTH),
                      char (WIDTH >> 8),
                      char (WIDTH >> 16),
                      char (WIDTH >> 24)};
   char biHeight[4] = {char (HEIGHT),
                       char (HEIGHT >> 8),
                       char (HEIGHT >> 16),
                       char (HEIGHT >> 24)};
                      
   char biPlanes[2]={0x01,0x00};
   char biBitCount[2]={0x18,0x00};
   char biCompression[4]={0x00,0x00,0x00,0x00};
   char biSizeImage[4]={0x00,0x00,0x00,0x00};
   char biXPelsPerMeter[4]={0x00,0x00,0x00,0x00};
   char biYPelsPerMeter[4]={0x00,0x00,0x00,0x00};
   char biClrUsed[4]={0x00,0x00,0x00,0x00};
   char biClrImportant[4]={0x00,0x00,0x00,0x00};

   char name[32];
   sprintf(name, "images/run_%04d/%0*d.bmp", run, len, num);

   fstream image;
   image.open (name, fstream::binary | fstream::out);
   
   image.put(bfType1);
   image.put(bfType2);
   for(int i=0;i<4;i++){image.put(bfSize[i]);}
   for(int i=0;i<2;i++){image.put(bfReserved1[i]);}
   for(int i=0;i<2;i++){image.put(bfReserved2[i]);}
   for(int i=0;i<4;i++){image.put(bfOffset[i]);}
   for(int i=0;i<4;i++){image.put(biSize[i]);}
   for(int i=0;i<4;i++){image.put(biWidth[i]);}
   for(int i=0;i<4;i++){image.put(biHeight[i]);}
   for(int i=0;i<2;i++){image.put(biPlanes[i]);}
   for(int i=0;i<2;i++){image.put(biBitCount[i]);}
   for(int i=0;i<4;i++){image.put(biCompression[i]);}
   for(int i=0;i<4;i++){image.put(biSizeImage[i]);}
   for(int i=0;i<4;i++){image.put(biXPelsPerMeter[i]);}
   for(int i=0;i<4;i++){image.put(biYPelsPerMeter[i]);}
   for(int i=0;i<4;i++){image.put(biClrUsed[i]);}
   for(int i=0;i<4;i++){image.put(biClrImportant[i]);}

   int offset = 4 - ((WIDTH * 3) % 4);
   if (offset == 4)
      offset = 0;
   
   for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
         uint32_t color = getColor(iters[i * WIDTH + j]);
         image.put(char(color >> 16));
         image.put(char(color >> 8));
         image.put(char(color));
      }
      
      for (int j = 0; j < offset; j++)
         image.put(0);
   }
   
   image.close();
}

inline bool BetterZoom(StdDevInfo_t *original, StdDevInfo_t *comp) {
   return     pow(comp->variance, 3.0) *     comp->numNonMax > 
          pow(original->variance, 3.0) * original->numNonMax;
}

double Variance(uint32_t iters[], double mean, uint32_t count) {
   if (count == 0)
      return 0.0;
   
   double sqrSum = 0.0;
   for (int i = 0; i < count; i++)
      sqrSum += pow(mean - (double)iters[i], 2);
   
   return (sqrSum/(count-1));
}

void insertSorted(StdDevInfo_t stdDevs[], int *varCount, uint32_t iters[], int count, StdDevInfo_t *curInfo) {
   if (count == 0)
      return;
   
   uint32_t sum = 0;
   int ndx = *varCount;
   
   for (int i = 0; i < count; i++)
      sum += iters[i];
   
   curInfo->mean = (double) sum / (double) count;
   curInfo->variance = Variance(iters, curInfo->mean, count);

   while (ndx > 0 && BetterZoom(&stdDevs[ndx - 1], curInfo)) {
      if (ndx < RANDOM_POOL_SIZE)
         stdDevs[ndx] = stdDevs[ndx - 1];
      ndx--;
   }
   
   if (ndx < RANDOM_POOL_SIZE) {
      if (*varCount < RANDOM_POOL_SIZE)
         ++*varCount;
      
      stdDevs[ndx] = *curInfo;
   }
}

void findPath(uint32_t *iters, int *xNdx, int *yNdx) {
   int count;
   uint32_t subIter[(2*STD_DEV_RADIUS+1)*(2*STD_DEV_RADIUS+1)];
   
   StdDevInfo_t stdDevs[RANDOM_POOL_SIZE];
   StdDevInfo_t curInfo;
   int varCount = 0;

   for (int i = STD_DEV_RADIUS; i < HEIGHT - 1 - STD_DEV_RADIUS; i++) {
      curInfo.yNdx = i;

      for (int j = STD_DEV_RADIUS; j < WIDTH - 1 - STD_DEV_RADIUS; j++) {
         curInfo.xNdx = j;

         count = 0;
         curInfo.numNonMax = 0;
         for (int k = i - STD_DEV_RADIUS; k <= i + STD_DEV_RADIUS; k++) {
            for (int l = j - STD_DEV_RADIUS; l <= j + STD_DEV_RADIUS; l++) {
               subIter[count++] = getColor(iters[k * WIDTH + l]);
               if (iters[k * WIDTH + l] != MAX) {
                  curInfo.numNonMax++;
               }
            }
         }
         
         insertSorted(stdDevs, &varCount, subIter, count, &curInfo);
      }
   }
   
   int path = clock() & (RANDOM_POOL_SIZE - 1);
   
   *xNdx = stdDevs[path].xNdx;
   *yNdx = stdDevs[path].yNdx;
}
