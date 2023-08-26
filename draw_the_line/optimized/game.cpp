#include "precomp.h"
#include "game.h"

#define V_MUTATION


#define LINES		1024
#define LINEFILE	"lines1024.dat"
#define ITERATIONS	16

#define IRand(x) ((int)(RandomFloat()*(x)))

int lx1[LINES], ly1[LINES], lx2[LINES], ly2[LINES];			// lines: start and end coordinates
int x1_, y1_, x2_, y2_;										// room for storing line backup
__int64 fitness = 0xfffffffff;								// similarity to reference image
int lidx = 0;												// current line to be mutated
float peak = 0;												// peak line rendering performance
Timer tm;													// stopwatch

uint8_t* optScreen;
uint8_t* optBackup;
uint8_t* optRef8;
uint8_t* optBestScreen;

#define PLOT(x, y, c) optScreen[(x) + (y) * SCRWIDTH] = (c)
#define READ(x, y) optScreen[(x) + (y) * SCRWIDTH]


constexpr std::array<float, 512> ComputeLookupWeights()
{
	std::array<float, 512> weights = {};
	constexpr float div = 1.f / 255.f;

	for (int i = 0; i < 256; ++i)
	{
		weights[i] = i * div;
		weights[256 + i] = (i ^ 255) * div;
	}

	return weights;
}

const static std::array<float, 512> lookupWeight = ComputeLookupWeights();

// -----------------------------------------------------------
// Mutate
// Randomly modify or replace one line.
// -----------------------------------------------------------
void MutateLine(int i)
{
#ifdef V_MUTATION
	static const __m128i maxV = _mm_set1_epi32(SCRWIDTH - 1);
	static const __m128i minV = _mm_set1_epi32(0);
#endif

	// backup the line before modifying it
	x1_ = lx1[i], y1_ = ly1[i];
	x2_ = lx2[i], y2_ = ly2[i];
	do
	{
		if (rand() & 1)
		{
			// small mutation (50% probability)
			lx1[i] += IRand(6) - 3, ly1[i] += IRand(6) - 3;
			lx2[i] += IRand(6) - 3, ly2[i] += IRand(6) - 3;
			// ensure the line stays on the screen

#ifdef V_MUTATION

			__m128i t;
			t.m128i_i32[0] = lx1[i];
			t.m128i_i32[1] = lx2[i];
			t.m128i_i32[2] = ly1[i];
			t.m128i_i32[3] = ly2[i];

			t = _mm_min_epi32(_mm_max_epi32(t, minV), maxV);

			lx1[i] = t.m128i_i32[0];
			lx2[i] = t.m128i_i32[1];
			ly1[i] = t.m128i_i32[2];
			ly2[i] = t.m128i_i32[3];
#else
			lx1[i] = min(SCRWIDTH - 1, max(0, lx1[i]));
			lx2[i] = min(SCRWIDTH - 1, max(0, lx2[i]));
			ly1[i] = min(SCRHEIGHT - 1, max(0, ly1[i]));
			ly2[i] = min(SCRHEIGHT - 1, max(0, ly2[i]));
#endif

		}
		else
		{
			// new line (50% probability)
			lx1[i] = IRand(SCRWIDTH), lx2[i] = IRand(SCRWIDTH);
			ly1[i] = IRand(SCRHEIGHT), ly2[i] = IRand(SCRHEIGHT);
		}
	} while ((abs(lx1[i] - lx2[i]) < 3) || (abs(ly1[i] - ly2[i]) < 3));
}

void UndoMutation(int i)
{
	// restore line i to the backuped state
	lx1[i] = x1_, ly1[i] = y1_;
	lx2[i] = x2_, ly2[i] = y2_;
}

// -----------------------------------------------------------
// DrawWuLine
// Anti-aliased line rendering.
// Straight from: 
// https://www.codeproject.com/Articles/13360/Antialiasing-Wu-Algorithm
// -----------------------------------------------------------
void DrawWuLine(int X0, int Y0, int X1, int Y1, BYTE clrLine)
{
	static union { int64_t t; int32_t r[2]; };
	r[0] = clrLine;

	/* Make sure the line runs top to bottom */
	if (Y0 > Y1)
	{
		std::swap(X0, X1);
		std::swap(Y0, Y1);
	}

	/* Draw the initial pixel, which is always exactly intersected by
	the line and so needs no weighting */
	PLOT(X0, Y0, clrLine);

	int XDir, DeltaX = X1 - X0;
	if (DeltaX >= 0)
	{
		XDir = 1;
	}
	else
	{
		XDir = -1;
		DeltaX = 0 - DeltaX; /* make DeltaX positive */
	}

	/* Special-case horizontal, vertical, and diagonal lines, which
	require no weighting because they go right through the center of
	every pixel */
	int DeltaY = Y1 - Y0;

	const int grayLine = clrLine;

	unsigned short ErrorAdj;
	unsigned short ErrorAccTemp, Weighting;

	/* Line is not horizontal, diagonal, or vertical */
	unsigned short ErrorAcc = 0;  /* initialize the line error accumulator to 0 */

	/* Is this an X-major or Y-major line? */
	if (DeltaY > DeltaX)
	{
		/* Y-major line; calculate 16-bit fixed-point fractional part of a
		pixel that X advances each time Y advances 1 pixel, truncating the
			result so that we won't overrun the endpoint along the X axis */
		ErrorAdj = ((unsigned long)DeltaX << 16) / (unsigned long)DeltaY;
		/* Draw all pixels other than the first and last */
		while (--DeltaY) {
			ErrorAccTemp = ErrorAcc;   /* remember currrent accumulated error */
			ErrorAcc += ErrorAdj;      /* calculate error for next pixel */
			if (ErrorAcc <= ErrorAccTemp) {
				/* The error accumulator turned over, so advance the X coord */
				X0 += XDir;
			}
			Y0++; /* Y-major, so always advance Y */
			/* The IntensityBits most significant bits of ErrorAcc give us the
			intensity weighting for this pixel, and the complement of the
	  weighting for the paired pixel */
			Weighting = ErrorAcc >> 8;

			const uint8_t ref1 = READ(X0, Y0);
			const uint8_t ref2 = READ(X0 + XDir, Y0);

			r[1] = ref1;
			const int16_t sub = r[1] - clrLine;
			const int16_t sign = (sub >> 31) & 1; // 0 -> >=, 1 -> < 
			const int16_t lookup = sign * 256 + Weighting;
			const BYTE color = lookupWeight[lookup] * std::abs(sub) + r[sign];

			r[1] = ref2;
			const int16_t sub2 = r[1] - clrLine;
			const int16_t sign2 = (sub2 >> 31) & 1; // 0 -> >=, 1 -> <  
			const int16_t lookup2 = (1 - sign2) * 256 + Weighting;
			const BYTE color2 = lookupWeight[lookup2] * std::abs(sub2) + r[sign2];

			PLOT(X0, Y0, color);
			PLOT(X0 + XDir, Y0, color2);

		}
		/* Draw the final pixel, which is always exactly intersected by the line
		and so needs no weighting */
		PLOT(X1, Y1, clrLine);
		return;
	}
	/* It's an X-major line; calculate 16-bit fixed-point fractional part of a
	pixel that Y advances each time X advances 1 pixel, truncating the
	result to avoid overrunning the endpoint along the X axis */
	ErrorAdj = ((unsigned long)DeltaY << 16) / (unsigned long)DeltaX;
	/* Draw all pixels other than the first and last */
	while (--DeltaX) {
		ErrorAccTemp = ErrorAcc;   /* remember currrent accumulated error */
		ErrorAcc += ErrorAdj;      /* calculate error for next pixel */
		if (ErrorAcc <= ErrorAccTemp) {
			/* The error accumulator turned over, so advance the Y coord */
			Y0++;
		}
		X0 += XDir; /* X-major, so always advance X */
		/* The IntensityBits most significant bits of ErrorAcc give us the
		intensity weighting for this pixel, and the complement of the
weighting for the paired pixel */
		Weighting = ErrorAcc >> 8;

		const uint8_t ref1 = READ(X0, Y0);
		const uint8_t ref2 = READ(X0, Y0 + 1);

		r[1] = ref1;
		const int sub = r[1] - clrLine;
		const int sign = (sub >> 31) & 1; // 0 -> >=, 1 -> <   
		const int lookup = sign * 256 + Weighting;
		const BYTE color = lookupWeight[lookup] * std::abs(sub) + r[sign];

		r[1] = ref2;
		const int sub2 = r[1] - clrLine;
		const int sign2 = (sub2 >> 31) & 1; // 0 -> >=, 1 -> <   
		const int lookup2 = (1 - sign2) * 256 + Weighting;
		const BYTE color2 = lookupWeight[lookup2] * std::abs(sub2) + r[sign2];

		PLOT(X0, Y0, color);
		PLOT(X0, Y0 + 1, color2);
	}

	/* Draw the final pixel, which is always exactly intersected by the line
	and so needs no weighting */
	PLOT(X1, Y1, clrLine);
}

// -----------------------------------------------------------
// Fitness evaluation
// Compare current generation against reference image.
// -----------------------------------------------------------

#define COMPUTE_FITNESS_CHUNK(a, b, mask, fitness)	{								\
	const __m128i d2 = _mm_abs_epi32(												\
		_mm_sub_epi32(_mm_and_si128(a, mask), _mm_and_si128(b, mask)));				\
	fitness = _mm_add_epi32(fitness, _mm_srai_epi32(_mm_mul_epi32(d2, d2), 12));	\
}																					\

#define COMPUTE_FITNESS_CHUNK_S(a, b, mask, fitness, shift)	{						\
	const __m128i d2 = _mm_abs_epi32(												\
		_mm_sub_epi32(_mm_srli_epi32(_mm_and_si128(a, mask), shift),				\
		_mm_srli_epi32(_mm_and_si128(b, mask), shift)));							\
	fitness = _mm_add_epi32(fitness, _mm_srai_epi32(_mm_mul_epi32(d2, d2), 12));	\
}			

__int64 Game::Evaluate()
{
	const static __m128i mask_1 = _mm_set1_epi32(0xff);
	const static __m128i mask_2 = _mm_set1_epi32(0xff00);
	const static __m128i mask_3 = _mm_set1_epi32(0xff0000);
	const static __m128i mask_4 = _mm_set1_epi32(0xff000000);


	const __m128i* A16 = (const __m128i*)optScreen;
	const __m128i* B16 = (const __m128i*)optRef8;

	union { __m128i diff4; int diff[4]; };
	diff4 = _mm_set1_epi32(0);


	for (int i = 0; i < (SCRWIDTH * SCRHEIGHT) / 16; ++i)
	{
		COMPUTE_FITNESS_CHUNK(A16[i], B16[i], mask_1, diff4);
		COMPUTE_FITNESS_CHUNK_S(A16[i], B16[i], mask_2, diff4, 8);
		COMPUTE_FITNESS_CHUNK_S(A16[i], B16[i], mask_3, diff4, 16);
		COMPUTE_FITNESS_CHUNK_S(A16[i], B16[i], mask_4, diff4, 24);
	}

	__int64 retval = diff[0];
	retval += diff[1];
	retval += diff[2];
	retval += diff[3];
	return retval;
}																		\

// -----------------------------------------------------------
// Application initialization
// Load a previously saved generation, if available.
// -----------------------------------------------------------
void Game::Init()
{
	for (int i = 0; i < LINES; i++) MutateLine(i);
	FILE* f = fopen(LINEFILE, "rb");
	if (f)
	{
		fread(lx1, 4, LINES, f);
		fread(ly1, 4, LINES, f);
		fread(lx2, 4, LINES, f);
		fread(ly2, 4, LINES, f);
		fclose(f);
	}
	Surface* reference = new Surface("assets/image3.png");
	fitness = 512 * 512 * 16;

	optScreen = (uint8_t*)MALLOC64(SCRWIDTH * SCRHEIGHT);
	optBackup = (uint8_t*)MALLOC64(SCRWIDTH * SCRHEIGHT);
	optRef8 = (uint8_t*)MALLOC64(SCRWIDTH * SCRHEIGHT);
	optBestScreen = (uint8_t*)MALLOC64(SCRWIDTH * SCRHEIGHT);

	for (int i = 0; i < SCRWIDTH * SCRHEIGHT; ++i) 
	{
		optRef8[i] = reference->pixels[i] & 255;
	}

}

// -----------------------------------------------------------
// Application termination
// Save the current generation, so we can continue later.
// -----------------------------------------------------------
void Game::Shutdown()
{
	FILE* f = fopen(LINEFILE, "wb");
	fwrite(lx1, 4, LINES, f);
	fwrite(ly1, 4, LINES, f);
	fwrite(lx2, 4, LINES, f);
	fwrite(ly2, 4, LINES, f);
	fclose(f);

	FREE64(optScreen);
	FREE64(optBackup);
	FREE64(optRef8);
	FREE64(optBestScreen);
}

// -----------------------------------------------------------
// Main application tick function
// -----------------------------------------------------------
void Game::Tick(float _DT)
{
	tm.reset();
	int lineCount = 0;
	int iterCount = 0;
	// draw up to lidx

	memset(optScreen, 255, SCRWIDTH * SCRHEIGHT);

	for (int j = 0; j < lidx; j++, lineCount++)
	{
		unsigned int c = j >> 3;
		DrawWuLine(lx1[j], ly1[j], lx2[j], ly2[j], c);
	}
	int base = lidx;
	memcpy(optBackup, optScreen, SCRWIDTH * SCRHEIGHT);

	bool updateScreen = false;

	// iterate and draw from lidx to end
	for (int k = 0; k < ITERATIONS; k++)
	{
		memcpy(optScreen, optBackup, SCRWIDTH * SCRHEIGHT);

		MutateLine(lidx);
		for (int j = base; j < LINES; j++, lineCount++)
		{
			unsigned int c = j >> 3;
			DrawWuLine(lx1[j], ly1[j], lx2[j], ly2[j], c);
		}

		const __int64 diff = Evaluate();

		if (diff < fitness) 
		{
			fitness = diff;
			updateScreen = true;
			memcpy(optBestScreen, optScreen, SCRWIDTH * SCRHEIGHT);
		}
		else
		{
			UndoMutation(lidx);
		}
		lidx = (lidx + 1) % LINES;
		iterCount++;
	}

	if (updateScreen)
	{
		for (int j = 0; j < SCRWIDTH * SCRHEIGHT; ++j)
		{
			const uint8_t grayValue = optBestScreen[j];
			screen->pixels[j] = RGB(grayValue, grayValue, grayValue);
		}
	}


	// stats
	char t[128];
	float elapsed = tm.elapsed();
	float lps = (float)lineCount / elapsed;
	peak = max(lps, peak);
	sprintf(t, "fitness: %i", fitness);
	screen->Bar(0, SCRHEIGHT - 33, 130, SCRHEIGHT - 1, 0);
	screen->Print(t, 2, SCRHEIGHT - 24, 0xffffff);
	sprintf(t, "lps:     %5.2fK", lps);
	screen->Print(t, 2, SCRHEIGHT - 16, 0xffffff);
	sprintf(t, "ips:     %5.2f", (iterCount * 1000) / elapsed);
	screen->Print(t, 2, SCRHEIGHT - 8, 0xffffff);
	sprintf(t, "peak:    %5.2f", peak);
	screen->Print(t, 2, SCRHEIGHT - 32, 0xffffff);
}