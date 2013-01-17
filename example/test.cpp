
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>

using namespace std;

double test(float* instance);

void clearinstance(float* instance, int numfeatures) {
    for (int f=0; f<numfeatures; f++)
        instance[f] = 0.f;
}

bool parseFeatureValue(string &cfeature, string &cvalue) {
    // get token
    char* tok;
    if (not (tok = strtok(NULL, " \n"))) // line stored in state from previous call to strtok
        return false;
    
    // tok is feature:value
	string bit = tok;
	int colon_index = bit.find(":");
	cfeature = bit.substr(0, colon_index);
	cvalue = bit.substr(colon_index+1,bit.length()-colon_index-1);
	
    return true;
}

bool readinstance(int numfeatures, float* instance) {
    // get line from stdin
    string strline;
	getline(cin, strline);
	char* line = strdup(strline.c_str());

	// extract and ignore label (first item)
	strtok(line, " ");

	// get qid, if present, and ignore
	string cfeature, cvalue;
    string qidstr ("qid");
    if (not parseFeatureValue(cfeature, cvalue)) return true;
    if (qidstr.compare(cfeature)) // qid is present
        if (not parseFeatureValue(cfeature, cvalue)) return true;
	
	// get feature values
    int feature = -1;
    float value = -1.f;
	do {
	    // parse and check feature index
	    feature = atoi(cfeature.c_str());
        if (feature < 0) return false;  // obviously invalid
        if (feature >= numfeatures) return true;
            // could be invalid, but most likely the trees found no use for features >= numfeatures
            // and we can skip the remaining feature values for this instance since they are expected
            // to be listed in ascending order by feature index
        
        // store feature value
		value = (float) atof(cvalue.c_str());
        instance[feature] = value;
    } while (parseFeatureValue(cfeature, cvalue));

	// clean up
	free(line);
	
	// return
    return true;
}

void driver(int numfeatures) {
    // variables
    float* instance = new float[numfeatures];
    clearinstance(instance, numfeatures);
    
	// evaluate all data instances
	while (readinstance(numfeatures, instance) and not cin.eof()) {
        // test instance
        double result = test(instance);
        
        // print result
        printf("%f\n", result);
        clearinstance(instance, numfeatures);
    }
}
double tree_0(float* instance) {if (instance[444] < 0.964970) {if (instance[349] < 0.147745) {if (instance[596] < 0.168410) {return 0.066667;} else {return 0.185714;}} else {if (instance[135] < 0.535605) {return 0.040000;} else {return 0.090625;}}} else {if (instance[8] < 0.950030) {return 0.300000;} else {if (instance[8] < 0.956305) {return 0.400000;} else {return 0.400000;}}}}
double tree_1(float* instance) {if (instance[444] < 0.964970) {if (instance[149] < 0.881890) {if (instance[637] < 0.272915) {return 0.092190;} else {return 0.040397;}} else {if (instance[36] < 0.734015) {return 0.150308;} else {return -0.005333;}}} else {if (instance[8] < 0.950030) {return 0.270000;} else {if (instance[8] < 0.956305) {return 0.360000;} else {return 0.360000;}}}}
double tree_2(float* instance) {if (instance[444] < 0.964970) {if (instance[349] < 0.147745) {if (instance[418] < 0.254210) {return 0.051790;} else {return 0.154643;}} else {if (instance[135] < 0.706510) {return 0.036924;} else {return 0.083809;}}} else {if (instance[8] < 0.950030) {return 0.243000;} else {if (instance[8] < 0.956305) {return 0.324000;} else {return 0.324000;}}}}
double tree_3(float* instance) {if (instance[444] < 0.964970) {if (instance[220] < 0.681520) {if (instance[349] < 0.147745) {return 0.132093;} else {return 0.060151;}} else {if (instance[585] < 0.475090) {return 0.076752;} else {return 0.000247;}}} else {if (instance[36] < 0.708395) {if (instance[8] < 0.956305) {return 0.291600;} else {return 0.291600;}} else {return 0.218700;}}}
double tree_4(float* instance) {if (instance[444] < 0.964970) {if (instance[149] < 0.852495) {if (instance[637] < 0.272915) {return 0.069496;} else {return 0.023680;}} else {if (instance[36] < 0.734015) {return 0.112272;} else {return 0.015997;}}} else {if (instance[8] < 0.950030) {return 0.196830;} else {if (instance[8] < 0.956305) {return 0.262440;} else {return 0.262440;}}}}
double tree_5(float* instance) {if (instance[100] < 0.909860) {if (instance[220] < 0.681520) {if (instance[300] < 0.626080) {return 0.039678;} else {return 0.089140;}} else {if (instance[43] < 0.672250) {return -0.019080;} else {return 0.035228;}}} else {if (instance[8] < 0.950030) {if (instance[8] < 0.779290) {return 0.126497;} else {return 0.177147;}} else {if (instance[8] < 0.956305) {return 0.236196;} else {return 0.236196;}}}}
double tree_6(float* instance) {if (instance[100] < 0.909860) {if (instance[669] < 0.260840) {if (instance[98] < 0.437070) {return 0.068832;} else {return -0.013733;}} else {if (instance[438] < 0.893625) {return 0.044656;} else {return 0.122340;}}} else {if (instance[17] < 0.342205) {if (instance[8] < 0.956305) {return 0.212576;} else {return 0.212576;}} else {if (instance[8] < 0.779290) {return 0.113847;} else {return 0.159432;}}}}
double tree_7(float* instance) {if (instance[100] < 0.909860) {if (instance[513] < 0.428760) {if (instance[596] < 0.266080) {return 0.059243;} else {return -0.024237;}} else {if (instance[607] < 0.753225) {return 0.053266;} else {return -0.000874;}}} else {if (instance[17] < 0.342205) {if (instance[8] < 0.956305) {return 0.191319;} else {return 0.191319;}} else {if (instance[8] < 0.779290) {return 0.102463;} else {return 0.143489;}}}}
double tree_8(float* instance) {if (instance[100] < 0.909860) {if (instance[220] < 0.681520) {if (instance[687] < 0.856730) {return 0.048503;} else {return -0.023691;}} else {if (instance[433] < 0.541990) {return 0.032745;} else {return -0.016562;}}} else {if (instance[8] < 0.950030) {if (instance[8] < 0.779290) {return 0.092216;} else {return 0.129140;}} else {if (instance[8] < 0.956305) {return 0.172187;} else {return 0.172187;}}}}
double tree_9(float* instance) {if (instance[100] < 0.909860) {if (instance[27] < 0.996140) {if (instance[433] < 0.604720) {return 0.039330;} else {return 0.008228;}} else {if (instance[242] < 0.386960) {return 0.042284;} else {return 0.101119;}}} else {if (instance[17] < 0.342205) {if (instance[8] < 0.956305) {return 0.154968;} else {return 0.154968;}} else {if (instance[8] < 0.779290) {return 0.082995;} else {return 0.116226;}}}}
double tree_10(float* instance) {if (instance[100] < 0.909860) {if (instance[513] < 0.428760) {if (instance[36] < 0.611805) {return -0.025127;} else {return 0.044535;}} else {if (instance[533] < 0.893380) {return 0.030954;} else {return 0.117501;}}} else {if (instance[8] < 0.950030) {if (instance[8] < 0.779290) {return 0.074695;} else {return 0.104604;}} else {if (instance[8] < 0.956305) {return 0.139471;} else {return 0.139471;}}}}
double tree_11(float* instance) {if (instance[100] < 0.909860) {if (instance[27] < 0.996140) {if (instance[36] < 0.988970) {return 0.018153;} else {return 0.134734;}} else {if (instance[433] < 0.861065) {return 0.087911;} else {return 0.034960;}}} else {if (instance[8] < 0.950030) {if (instance[8] < 0.779290) {return 0.067226;} else {return 0.094143;}} else {if (instance[8] < 0.956305) {return 0.125524;} else {return 0.125524;}}}}
double tree_12(float* instance) {if (instance[100] < 0.909860) {if (instance[669] < 0.260840) {if (instance[98] < 0.437070) {return 0.043474;} else {return -0.020572;}} else {if (instance[190] < 0.829825) {return 0.037245;} else {return -0.017906;}}} else {if (instance[8] < 0.950030) {if (instance[8] < 0.779290) {return 0.060503;} else {return 0.084729;}} else {if (instance[8] < 0.956305) {return 0.112972;} else {return 0.112972;}}}}
double tree_13(float* instance) {if (instance[100] < 0.909860) {if (instance[27] < 0.996140) {if (instance[505] < 0.995055) {return 0.014086;} else {return 0.118704;}} else {if (instance[433] < 0.861065) {return 0.075396;} else {return 0.033388;}}} else {if (instance[8] < 0.950030) {if (instance[8] < 0.779290) {return 0.054453;} else {return 0.076256;}} else {if (instance[8] < 0.956305) {return 0.101675;} else {return 0.101675;}}}}
double tree_14(float* instance) {if (instance[220] < 0.681520) {if (instance[687] < 0.861550) {if (instance[179] < 0.851025) {return 0.028019;} else {return 0.086121;}} else {if (instance[108] < 0.605310) {return -0.018695;} else {return -0.030285;}}} else {if (instance[433] < 0.541990) {if (instance[8] < 0.881260) {return 0.035920;} else {return -0.014281;}} else {if (instance[124] < 0.933305) {return -0.027694;} else {return 0.029999;}}}}
double tree_15(float* instance) {if (instance[253] < 0.892320) {if (instance[34] < 0.804855) {if (instance[433] < 0.634240) {return 0.017815;} else {return -0.020797;}} else {if (instance[276] < 0.772255) {return 0.017434;} else {return 0.049383;}}} else {if (instance[248] < 0.930910) {if (instance[8] < 0.784930) {return 0.059108;} else {return 0.065828;}} else {if (instance[36] < 0.717010) {return 0.082895;} else {return 0.078393;}}}}
double tree_16(float* instance) {if (instance[220] < 0.681520) {if (instance[359] < 0.973465) {if (instance[122] < 0.607320) {return 0.037542;} else {return 0.008719;}} else {return -0.076798;}} else {if (instance[433] < 0.541990) {if (instance[8] < 0.881260) {return 0.030556;} else {return -0.014616;}} else {if (instance[124] < 0.933305) {return -0.024766;} else {return 0.022060;}}}}
double tree_17(float* instance) {if (instance[100] < 0.909860) {if (instance[693] < 0.433710) {if (instance[36] < 0.611805) {return -0.025696;} else {return 0.022664;}} else {if (instance[533] < 0.893380) {return 0.015231;} else {return 0.085096;}}} else {if (instance[8] < 0.950030) {if (instance[8] < 0.779290) {return 0.040395;} else {return 0.055491;}} else {if (instance[8] < 0.956305) {return 0.077397;} else {return 0.070851;}}}}
double tree_18(float* instance) {if (instance[253] < 0.892320) {if (instance[36] < 0.988970) {if (instance[34] < 0.804855) {return -0.001175;} else {return 0.022201;}} else {return 0.100457;}} else {if (instance[248] < 0.930910) {if (instance[8] < 0.784930) {return 0.047919;} else {return 0.049942;}} else {if (instance[36] < 0.717010) {return 0.063766;} else {return 0.065277;}}}}
double tree_19(float* instance) {if (instance[216] < 0.041857) {if (instance[216] < 0.022978) {if (instance[32] < 0.666500) {return -0.026733;} else {return -0.019110;}} else {return -0.040971;}} else {if (instance[179] < 0.851025) {if (instance[149] < 0.591720) {return -0.001820;} else {return 0.025135;}} else {if (instance[124] < 0.423775) {return 0.071746;} else {return 0.025884;}}}}
double tree_20(float* instance) {if (instance[359] < 0.973465) {if (instance[220] < 0.681520) {if (instance[399] < 0.864565) {return 0.021274;} else {return -0.023342;}} else {if (instance[433] < 0.541990) {return 0.011501;} else {return -0.019861;}}} else {return -0.070341;}}
double test(float* instance) {
	double pred = 0.f;
	pred += tree_0(instance);
	pred += tree_1(instance);
	pred += tree_2(instance);
	pred += tree_3(instance);
	pred += tree_4(instance);
	pred += tree_5(instance);
	pred += tree_6(instance);
	pred += tree_7(instance);
	pred += tree_8(instance);
	pred += tree_9(instance);
	pred += tree_10(instance);
	pred += tree_11(instance);
	pred += tree_12(instance);
	pred += tree_13(instance);
	pred += tree_14(instance);
	pred += tree_15(instance);
	pred += tree_16(instance);
	pred += tree_17(instance);
	pred += tree_18(instance);
	pred += tree_19(instance);
	pred += tree_20(instance);
	return pred;
}
int main(int argc, char* argv[]) { driver(694); }
