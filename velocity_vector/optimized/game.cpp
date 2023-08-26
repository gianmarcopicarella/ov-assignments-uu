#include "precomp.h"
#include "game.h"

// Profiling dependencies
#include <chrono>

#define GRIDSIZE 256

// VERLET CLOTH SIMULATION DEMO
// High-level concept: a grid consists of points, each connected to four 
// neighbours. For a simulation step, the position of each point is affected
// by its speed, expressed as (current position - previous position), a
// constant gravity force downwards, and random impulses ("wind").
// The final force is provided by the bonds between points, via the four
// connections.
// Together, this simple scheme yields a pretty convincing cloth simulation.
// The algorithm has been used in games since the game "Thief".

// ASSIGNMENT STEPS:
// 1. SIMD, part 1: in Game::Simulation, convert lines 119 to 126 to SIMD.
//    You receive 2 points if the resulting code is faster than the original.
//    This will probably require a reorganization of the data layout, which
//    may in turn require changes to the rest of the code.
// 2. SIMD, part 2: for an additional 4 points, convert the full Simulation
//    function to SSE. This may require additional changes to the data to
//    avoid concurrency issues when operating on neighbouring points.
//    The resulting code must be at least 2 times faster (using SSE) or 4
//    times faster (using AVX) than the original  to receive the full 4 points.
// 3. GPGPU, part 1: modify Game::Simulation so that it sends the cloth data
//    to the GPU, and execute lines 119 to 126 on the GPU. After this, bring
//    back the cloth data to the CPU and execute the remainder of the Verlet
//    simulation code. You receive 2 points if the code *works* correctly;
//    note that this is expected to be slower due to the data transfers.
// 4. GPGPU, part 2: execute the full Game::Simulation function on the GPU.
//    You receive 4 additional points if this yields a correct simulation
//    that is at least 5x faster than the original code. DO NOT draw the
//    cloth on the GPU; this is (for now) outside the scope of the assignment.
// Note that the GPGPU tasks will benefit from the SIMD tasks.
// Also note that your final grade will be capped at 10.

//#define PROFILE
//#define SIMD

#ifdef SIMD

constexpr int ELEMENTS_PER_ROW = 128;
constexpr int QUADS_PER_ROW = ELEMENTS_PER_ROW / 2;

float2* positionsOdd = (float2*)MALLOC64(GRIDSIZE * ELEMENTS_PER_ROW * sizeof(float2));
float2* positionsEven = (float2*)MALLOC64(GRIDSIZE * ELEMENTS_PER_ROW * sizeof(float2));

float2* positionsPrevOdd = (float2*)MALLOC64(GRIDSIZE * ELEMENTS_PER_ROW * sizeof(float2));
float2* positionsPrevEven = (float2*)MALLOC64(GRIDSIZE * ELEMENTS_PER_ROW * sizeof(float2));

float2* fixPositionsOdd = (float2*)MALLOC64(ELEMENTS_PER_ROW * sizeof(float2));
float2* fixPositionsEven = (float2*)MALLOC64(ELEMENTS_PER_ROW * sizeof(float2));

constexpr int CONST_CEIL(float aNum)
{
	return (static_cast<float>(static_cast<int32_t>(aNum)) == aNum)
		? static_cast<int32_t>(aNum)
		: static_cast<int32_t>(aNum) + ((aNum > 0) ? 1 : 0);
}

constexpr int GET_REST_ELEMENTS_COUNT()
{
	constexpr int elementsCount = (GRIDSIZE - 2) * ELEMENTS_PER_ROW;
	constexpr float groups = static_cast<float>(elementsCount) / 4.f;
	constexpr int perfectGroupsCount = CONST_CEIL(groups) * 16;

	return perfectGroupsCount;
}

constexpr int REST_ELEMENTS_COUNT = GET_REST_ELEMENTS_COUNT();

float* oddRestLengths = (float*)MALLOC64(REST_ELEMENTS_COUNT * sizeof(float));
float* evenRestLengths = (float*)MALLOC64(REST_ELEMENTS_COUNT * sizeof(float));

/* Vectorized arrays */
__m128* positionsOdd4 = (__m128*)positionsOdd;
__m128* prevPositionsOdd4 = (__m128*)positionsPrevOdd;
__m128* rightOdd4 = (__m128*)(positionsEven + 1);
__m128* leftOdd4 = (__m128*)positionsEven;
__m128* bottomOdd4 = (__m128*)(positionsOdd + ELEMENTS_PER_ROW);
__m128* topOdd4 = (__m128*)(positionsOdd - ELEMENTS_PER_ROW);
const __m128* restLengthsOdd4 = (const __m128*)oddRestLengths;

__m128* positionsEven4 = (__m128*)positionsEven;
__m128* prevPositionsEven4 = (__m128*)positionsPrevEven;
__m128* rightEven4 = (__m128*)positionsOdd;
__m128* leftEven4 = (__m128*)(positionsOdd - 1);
__m128* bottomEven4 = (__m128*)(positionsEven + ELEMENTS_PER_ROW);
__m128* topEven4 = (__m128*)(positionsEven - ELEMENTS_PER_ROW);
const __m128* restLengthsEven4 = (const __m128*)evenRestLengths;
/* End vectorized arrays */

// initialization
void Game::Init()
{
	for (int y = 0; y < GRIDSIZE; y++)
	{
		for (int x = 0; x < GRIDSIZE; x += 2)
		{
			int i = y * ELEMENTS_PER_ROW + x / 2;

			positionsEven[i].x = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
			positionsEven[i].y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);

			positionsOdd[i].x = 10 + (float)(x + 1) * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
			positionsOdd[i].y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);

			positionsPrevEven[i] = positionsEven[i];
			positionsPrevOdd[i] = positionsOdd[i];
		}
	}

	for (int i = 0; i < ELEMENTS_PER_ROW; ++i)
	{
		fixPositionsOdd[i] = positionsOdd[i];
		fixPositionsEven[i] = positionsEven[i];
	}

	// Rest lengths
	// Odd and Even (Right, Left, Bottom, Top)
	std::fill(oddRestLengths, oddRestLengths + REST_ELEMENTS_COUNT, std::numeric_limits<float>::infinity());
	std::fill(evenRestLengths, evenRestLengths + REST_ELEMENTS_COUNT, std::numeric_limits<float>::infinity());

	for (int i = ELEMENTS_PER_ROW, k = 0, j = 0; i < (GRIDSIZE - 1) * ELEMENTS_PER_ROW; ++i)
	{
		if (i % ELEMENTS_PER_ROW != (ELEMENTS_PER_ROW - 1))
		{
			oddRestLengths[k] = length(positionsOdd[i] - positionsEven[i + 1]) * 1.15f;
			oddRestLengths[k + 4] = length(positionsOdd[i] - positionsEven[i]) * 1.15f;
			oddRestLengths[k + 8] = length(positionsOdd[i] - positionsOdd[i + ELEMENTS_PER_ROW]) * 1.15f;
			oddRestLengths[k + 12] = length(positionsOdd[i] - positionsOdd[i - ELEMENTS_PER_ROW]) * 1.15f;
		}

		if (i % ELEMENTS_PER_ROW != 0)
		{
			evenRestLengths[j] = length(positionsEven[i] - positionsOdd[i]) * 1.15f;
			evenRestLengths[j + 4] = length(positionsEven[i] - positionsOdd[i - 1]) * 1.15f;
			evenRestLengths[j + 8] = length(positionsEven[i] - positionsEven[i + ELEMENTS_PER_ROW]) * 1.15f;
			evenRestLengths[j + 12] = length(positionsEven[i] - positionsEven[i - ELEMENTS_PER_ROW]) * 1.15f;
		}

		if ((++k) % 4 == 0)
		{
			k += 12;
		}

		if ((++j) % 4 == 0)
		{
			j += 12;
		}
	}
}

void Game::Shutdown()
{
	FREE64(positionsOdd);
	FREE64(positionsEven);
	FREE64(positionsPrevEven);
	FREE64(positionsPrevOdd);
	FREE64(fixPositionsOdd);
	FREE64(fixPositionsEven);
	FREE64(oddRestLengths);
	FREE64(evenRestLengths);
}

// cloth rendering
// NOTE: For this assignment, please do not attempt to render directly on
// the GPU. Instead, if you use GPGPU, retrieve simulation results each frame
// and render using the function below. Do not modify / optimize it.

#define DRAW_LINE3(p1, p2, p3)							\
	screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);		\
	screen->Line(p1.x, p1.y, p3.x, p3.y, 0xffffff)		\

#define DRAW_LINE2(p1, p2)								\
	screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff)		\

void Game::DrawGrid()
{
	// draw the grid
	screen->Clear(0);

	for (int i = 0; i < (GRIDSIZE - 1) * ELEMENTS_PER_ROW; ++i)
	{
		if (i % ELEMENTS_PER_ROW == 127)
		{
			DRAW_LINE2(
				positionsEven[i],
				positionsEven[i + ELEMENTS_PER_ROW]);
			continue;
		}

		DRAW_LINE3(positionsOdd[i], positionsEven[i + 1], positionsOdd[i + ELEMENTS_PER_ROW]);

		if (i % ELEMENTS_PER_ROW == 126)
		{
			continue;
		}

		DRAW_LINE3(positionsEven[i + 1], positionsOdd[i + 1], positionsEven[i + 1 + ELEMENTS_PER_ROW]);
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.13f;

#ifdef PROFILE
static long long int samples_task_count = 1;
static long long int avg_task1 = 0;
static long long int avg_task2 = 0;
static bool profile_print_flag = false;
#endif

inline void mm_solve_constraints(__m128& n1, __m128& n2, __m128& p1, __m128& p2, const __m128& rl)
{
	static const __m128 half_one = _mm_set_ps1(0.5f);
	static const __m128 zero = _mm_setzero_ps();
	static const __m128 inf = _mm_set_ps1(std::numeric_limits<float>::infinity());
	static const __m128 nan = _mm_set_ps1(std::numeric_limits<float>::quiet_NaN());

	const __m128 sub1 = _mm_sub_ps(n1, p1);
	const __m128 sub2 = _mm_sub_ps(n2, p2);
	const __m128 sub1_square = _mm_mul_ps(sub1, sub1);
	const __m128 sub2_square = _mm_mul_ps(sub2, sub2);
	const __m128 dist = _mm_sqrt_ps(_mm_hadd_ps(sub1_square, sub2_square));
	const __m128 MASK = _mm_and_ps(
		_mm_and_ps(_mm_cmpneq_ps(dist, inf), _mm_cmpneq_ps(dist, nan)),
		_mm_cmpgt_ps(dist, rl)
	);
	const __m128 extra = _mm_and_ps(
		_mm_mul_ps(
			_mm_div_ps(
				_mm_sub_ps(dist, rl),
				rl),
			half_one),
		MASK
	);
	const __m128 extra1 = _mm_unpacklo_ps(extra, extra);
	const __m128 extra2 = _mm_unpackhi_ps(extra, extra);
	const __m128 dir1 = _mm_and_ps(_mm_cmpneq_ps(extra1, zero), _mm_mul_ps(sub1, extra1));
	const __m128 dir2 = _mm_and_ps(_mm_cmpneq_ps(extra2, zero), _mm_mul_ps(sub2, extra2));

	p1 = _mm_add_ps(p1, dir1);
	p2 = _mm_add_ps(p2, dir2);
	n1 = _mm_sub_ps(n1, dir1);
	n2 = _mm_sub_ps(n2, dir2);
}

inline __m128i mm_rand_epu31()
{
	const static __m128i mask = _mm_set1_epi32(0x7fffffff);
	static __m128i seed = _mm_set_epi32(0x12345678, 0x56781234, 0x87651234, 0x87634512);

	seed = _mm_xor_si128(seed, _mm_slli_epi32(seed, 13));
	seed = _mm_xor_si128(seed, _mm_srli_epi32(seed, 17));
	seed = _mm_xor_si128(seed, _mm_slli_epi32(seed, 5));

	return _mm_and_si128(seed, mask);
}

inline void mm_verlet(const int i, __m128* pos, __m128* prevPos, const __m128& ACoeff)
{
	static const __m128 gravity = _mm_setr_ps(0, 0.003f, 0, 0.003f);
	const static __m128i maskT = _mm_setr_epi32(0xffffffff, 0, 0xffffffff, 0);
	const static __m128i maskC = _mm_setr_epi32(0, 0xffffffff, 0, 0xffffffff);
	static const __m128i ATresh = _mm_setr_epi32(6442450, 0, 6442450, 0);

	// Rand section
	const __m128i random128 = mm_rand_epu31();
	const __m128i computeMask = _mm_cmpgt_epi64(ATresh, _mm_and_epi32(random128, maskT));
	const __m128i K = _mm_and_epi32(computeMask, _mm_and_epi32(random128, maskC));
	const __m128i J = _mm_or_epi32(_mm_srli_epi64(K, 32), K);

	float* const ptr = (float*)(pos + i);
	float* const pptr = (float*)(prevPos + i);

	const __m128 curpos = _mm_load_ps(ptr);

	_mm_store_ps(ptr,
		_mm_add_ps(
			_mm_add_ps(
				_mm_mul_ps(_mm_cvtepi32_ps(J), ACoeff),
				_mm_add_ps(_mm_sub_ps(curpos, _mm_load_ps(pptr)), gravity)
			),
			curpos
		)
	);

	_mm_store_ps(pptr, curpos);
}

void Game::Simulation()
{
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::nanoseconds;

	const static float normalizationC = 4.6566128752457e-10f;
	static __m128 ACoeff = _mm_setr_ps(0, normalizationC * 0.12f, 0, normalizationC * 0.12f);

	// simulation is exected three times per frame; do not change this.
	for (int steps = 0; steps < 3; steps++)
	{
		// verlet integration; apply gravity
#ifdef PROFILE
		const auto t1_task1 = high_resolution_clock::now();
#endif

		ACoeff.m128_f32[0] = normalizationC * magic;
		ACoeff.m128_f32[2] = normalizationC * magic;

		for (int i = 0; i < GRIDSIZE * QUADS_PER_ROW; ++i)
		{
			mm_verlet(i, positionsEven4, prevPositionsEven4, ACoeff);
			mm_verlet(i, positionsOdd4, prevPositionsOdd4, ACoeff);
		}

#ifdef PROFILE
		const auto t2_task1 = high_resolution_clock::now();
		const auto ms_int1 = duration_cast<nanoseconds>(t2_task1 - t1_task1);
		avg_task1 = avg_task1 + (ms_int1.count() - avg_task1) / samples_task_count;
#endif

		magic += 0.0002f; // slowly increases the chance of anomalies
		// apply constraints; 4 simulation steps: do not change this number.

#ifdef PROFILE
		const auto t1_task2 = high_resolution_clock::now();
#endif

		for (int i = 0; i < 4; i++)
		{
			int k = 0, j = 0;

			for (int y = 1; y < GRIDSIZE - 1; y++)
			{
				for (int i = y * QUADS_PER_ROW; i < (y + 1) * QUADS_PER_ROW; i += 2, j += 4)
				{
					__m128 p1 = positionsOdd4[i];
					__m128 p2 = positionsOdd4[i + 1];

					__m128& r1 = rightOdd4[i];
					__m128& r2 = rightOdd4[i + 1];
					__m128& l1 = leftOdd4[i];
					__m128& l2 = leftOdd4[i + 1];
					__m128& b1 = bottomOdd4[i];
					__m128& b2 = bottomOdd4[i + 1];
					__m128& t1 = topOdd4[i];
					__m128& t2 = topOdd4[i + 1];

					const __m128& restR = restLengthsOdd4[j];
					const __m128& restL = restLengthsOdd4[j + 1];
					const __m128& restB = restLengthsOdd4[j + 2];
					const __m128& restT = restLengthsOdd4[j + 3];

					mm_solve_constraints(r1, r2, p1, p2, restR);
					mm_solve_constraints(l1, l2, p1, p2, restL);
					mm_solve_constraints(b1, b2, p1, p2, restB);
					mm_solve_constraints(t1, t2, p1, p2, restT);

					positionsOdd4[i] = p1;
					positionsOdd4[i + 1] = p2;
				}

				for (int i = y * QUADS_PER_ROW; i < (y + 1) * QUADS_PER_ROW; i += 2, k += 4)
				{
					__m128 p1 = positionsEven4[i];
					__m128 p2 = positionsEven4[i + 1];

					__m128& r1 = rightEven4[i];
					__m128& r2 = rightEven4[i + 1];
					__m128& l1 = leftEven4[i];
					__m128& l2 = leftEven4[i + 1];
					__m128& b1 = bottomEven4[i];
					__m128& b2 = bottomEven4[i + 1];
					__m128& t1 = topEven4[i];
					__m128& t2 = topEven4[i + 1];

					const __m128& restR = restLengthsEven4[k];
					const __m128& restL = restLengthsEven4[k + 1];
					const __m128& restB = restLengthsEven4[k + 2];
					const __m128& restT = restLengthsEven4[k + 3];

					mm_solve_constraints(r1, r2, p1, p2, restR);
					mm_solve_constraints(l1, l2, p1, p2, restL);
					mm_solve_constraints(b1, b2, p1, p2, restB);
					mm_solve_constraints(t1, t2, p1, p2, restT);

					positionsEven4[i] = p1;
					positionsEven4[i + 1] = p2;
				}
			}

			// fixed line of points is fixed.
			memcpy(positionsEven, fixPositionsEven, ELEMENTS_PER_ROW * sizeof(float2));
			memcpy(positionsOdd, fixPositionsOdd, ELEMENTS_PER_ROW * sizeof(float2));
		}

#ifdef PROFILE
		const auto t2_task2 = high_resolution_clock::now();
		const auto ms_int2 = duration_cast<nanoseconds>(t2_task2 - t1_task2);
		avg_task2 = avg_task2 + (ms_int2.count() - avg_task2) / samples_task_count;

		++samples_task_count;
#endif
	}

#ifdef PROFILE
	if (samples_task_count > 999 && !profile_print_flag)
	{
		std::cout << "Task 1: " << avg_task1 << " ns" << std::endl;
		std::cout << "Task 2: " << avg_task2 << " ns" << std::endl;

		profile_print_flag = true;
	}
#endif

}

#else // GPGPU

int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

// Our data layout
constexpr auto WIDTH = GRIDSIZE;
constexpr auto HEIGHT = GRIDSIZE;
constexpr auto ITEMS_COUNT = WIDTH * HEIGHT;

float2* positions = (float2*)MALLOC64(ITEMS_COUNT * sizeof(float2));

#define POS_REF(x, y) positions[(x) + (y) * WIDTH]			

// OpenCL Data
static Kernel* integrationKernel = nullptr;
static Kernel* constraintsKernel = nullptr;
static Buffer* positionsCl = nullptr;
static Buffer* prevPositionsCl = nullptr;
static Buffer* restLengthsCl = nullptr;
static Buffer* indicesCl = nullptr;

std::vector<int> computeDiagonalIndices()
{
	std::vector<int> indices;
	indices.resize(ITEMS_COUNT * 2);
	std::fill(indices.begin(), indices.end(), -1);

	for (int k = 0; k < GRIDSIZE * 2; k++)
	{
		for (int y = 0, idx = k * GRIDSIZE; y <= k; y++)
		{
			const int x = k - y;
			if (x > 0 && y > 0 && x < (GRIDSIZE - 1) && y < (GRIDSIZE - 1))
			{
				indices[idx++] = x + y * GRIDSIZE;
			}
		}
	}

	return indices;
}

void Game::Init()
{
	// create the cloth
	for (int y = 0; y < GRIDSIZE; y++)
	{
		for (int x = 0; x < GRIDSIZE; x++)
		{
			POS_REF(x, y).x = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
			POS_REF(x, y).y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);
		}
	}

	std::vector<float> restLengths;

	for (int y = 0; y < GRIDSIZE; ++y)
	{
		for (int x = 0; x < GRIDSIZE; ++x)
		{
			if (x > 0 && x < (GRIDSIZE - 1) && y > 0 && y < (GRIDSIZE - 1))
			{
				// calculate and store distance to four neighbours, allow 15% slack
				const auto currentPos = POS_REF(x, y);
				for (int c = 0; c < 4; c++)
				{
					const auto neighbourPos = POS_REF(x + xoffset[c], y + yoffset[c]);
					restLengths.push_back(length(currentPos - neighbourPos) * 1.15f);
				}
			}
			else
			{
				restLengths.push_back(0.f);
				restLengths.push_back(0.f);
				restLengths.push_back(0.f);
				restLengths.push_back(0.f);
			}
		} 
	}

	if (!integrationKernel)
	{
		Kernel::InitCL();

		integrationKernel = new Kernel("cl/verlet.cl", "integrate");
		constraintsKernel = new Kernel("cl/constraints.cl", "solve");

		// Positions
		positionsCl = new Buffer(ITEMS_COUNT * sizeof(float2), positions, Buffer::DEFAULT);

		// PrevPositions
		std::vector<float2> prevPositions;
		prevPositions.resize(ITEMS_COUNT);
		std::copy(positions, positions + ITEMS_COUNT, prevPositions.begin());
		prevPositionsCl = new Buffer(ITEMS_COUNT * sizeof(float2), &prevPositions[0], Buffer::DEFAULT);

		// Seeds
		std::vector<uint> seeds;
		seeds.push_back(0x12345678);
		for (int i = 1; i < ITEMS_COUNT; ++i) seeds.push_back(RandomUInt());
		Buffer* seedsCl = new Buffer(ITEMS_COUNT * sizeof(uint), &seeds[0], Buffer::DEFAULT);

		// Magics
		constexpr float magic = 0.13f;
		std::vector<float> magics;
		for (int i = 0; i < ITEMS_COUNT; ++i) magics.push_back(magic);
		Buffer* magicsCl = new Buffer(sizeof(float) * ITEMS_COUNT, &magics[0], Buffer::READONLY);

		integrationKernel->SetArguments(positionsCl, prevPositionsCl, seedsCl, magicsCl);

		magicsCl->CopyToDevice();
		seedsCl->CopyToDevice();
		positionsCl->CopyToDevice();
		prevPositionsCl->CopyToDevice();

		// Rest Lengths
		restLengthsCl = new Buffer(ITEMS_COUNT * 4 * sizeof(float), &restLengths[0], Buffer::READONLY);

		// Diagonal indices
		auto& indices = computeDiagonalIndices();
		indicesCl = new Buffer(2 * ITEMS_COUNT * sizeof(int), &indices[0], Buffer::READONLY);

		constraintsKernel->SetArguments(positionsCl, restLengthsCl, indicesCl);

		indicesCl->CopyToDevice();
		restLengthsCl->CopyToDevice();
	}
}

void Game::Shutdown()
{
	FREE64(positions);
}

// cloth rendering
// NOTE: For this assignment, please do not attempt to render directly on
// the GPU. Instead, if you use GPGPU, retrieve simulation results each frame
// and render using the function below. Do not modify / optimize it.
void Game::DrawGrid()
{
	positionsCl->CopyFromDevice();

	// draw the grid
	screen->Clear(0);
	for (int y = 0; y < (GRIDSIZE - 1); y++) for (int x = 1; x < (GRIDSIZE - 2); x++)
	{
		const float2 p1 = POS_REF(x, y);
		const float2 p2 = POS_REF(x + 1, y);
		const float2 p3 = POS_REF(x, y + 1);
		screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);
		screen->Line(p1.x, p1.y, p3.x, p3.y, 0xffffff);
	}
	for (int y = 0; y < (GRIDSIZE - 1); y++)
	{
		const float2 p1 = POS_REF(GRIDSIZE - 2, y);
		const float2 p2 = POS_REF(GRIDSIZE - 2, y + 1);
		screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);
	}
}


// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).

#ifdef PROFILE
static long long int samples_task_count = 1;
static long long int avg_task1 = 0;
static long long int avg_task2 = 0;
static bool profile_print_flag = false;
#endif

void Game::Simulation()
{
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::nanoseconds;

	static const int2 groupSize{ 256, 1 };

	// simulation is exected three times per frame; do not change this.
	for (int steps = 0; steps < 3; steps++)
	{
		// verlet integration; apply gravity

#ifdef PROFILE
		const auto t1_task1 = high_resolution_clock::now();
#endif

		integrationKernel->Run(ITEMS_COUNT);

#ifdef PROFILE
		const auto t2_task1 = high_resolution_clock::now();
		const auto ms_int1 = duration_cast<nanoseconds>(t2_task1 - t1_task1);
		avg_task1 = avg_task1 + (ms_int1.count() - avg_task1) / samples_task_count;
#endif

		// apply constraints; 4 simulation steps: do not change this number.
#ifdef PROFILE
		const auto t1_task2 = high_resolution_clock::now();
#endif

		constraintsKernel->Run2D(groupSize, groupSize);
		constraintsKernel->Run2D(groupSize, groupSize);
		constraintsKernel->Run2D(groupSize, groupSize);
		constraintsKernel->Run2D(groupSize, groupSize);

#ifdef PROFILE
		const auto t2_task2 = high_resolution_clock::now();
		const auto ms_int2 = duration_cast<nanoseconds>(t2_task2 - t1_task2);
		avg_task2 = avg_task2 + (ms_int2.count() - avg_task2) / samples_task_count;

		++samples_task_count;
#endif
	}

#ifdef PROFILE
	if (samples_task_count > 999 && !profile_print_flag)
	{
		std::cout << "Task 1: " << avg_task1 << " ns" << std::endl;
		std::cout << "Task 2: " << avg_task2 << " ns" << std::endl;

		profile_print_flag = true;
	}
#endif
}

#endif

void Game::Tick(float a_DT)
{
	// update the simulation
	Timer tm;
	tm.reset();
	Simulation();
	float elapsed1 = tm.elapsed();

	// draw the grid
	tm.reset();
	DrawGrid();
	float elapsed2 = tm.elapsed();

	// display statistics
	char t[128];
	sprintf(t, "ye olde ruggeth cloth simulation: %5.1f ms", elapsed1 * 1000);
	screen->Print(t, 2, SCRHEIGHT - 24, 0xffffff);
	sprintf(t, "                       rendering: %5.1f ms", elapsed2 * 1000);
	screen->Print(t, 2, SCRHEIGHT - 14, 0xffffff);
}