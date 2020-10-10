/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>  // std::istringstream
#include <string>   // std::string
#include <cstring>  // strlen and strcpy

#include <cinttypes>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <libgen.h>
#include <sys/stat.h>

#include <fmt/format.h>

class Candidate {
public:
    float        snr;
    int64_t      sample_idx;
    float        sample_time;
    unsigned int filter;
    unsigned int dm_trial;
    float        dm;
    unsigned int members;
    int64_t      begin;
    int64_t      end;
    unsigned     nbeams;
    unsigned     beam_mask;
    unsigned int primary_beam;
    float        max_snr;
    unsigned int beam;

    Candidate() {
        snr = sample_idx = sample_time = filter = dm_trial = dm = members = 0;
        begin = end = nbeams = beam_mask = primary_beam = max_snr = 0;
    }

    Candidate(const char* line, unsigned _beam_number) {
        std::istringstream iss(line, std::istringstream::in);
        iss >> snr;
        iss >> sample_idx;
        iss >> sample_time;
        iss >> filter;
        iss >> dm_trial;
        iss >> dm;
        iss >> members;
        iss >> begin;
        iss >> end;

        nbeams       = 1;
        beam_mask    = 1 << (_beam_number - 1);
        primary_beam = _beam_number;
        beam         = _beam_number;
        max_snr      = snr;

        iss >> ws;

        if (!iss.eof()){
            fmt::print(
                "Candiate::Candidate too many params on input line [{}]\n", 
                line);
        }
    }

    ~Candidate() {}

    void header() {
        fmt::print("SNR\tsamp_idx\ttime\tfilter\tdm_trial\tDM\tmembers\t"
                   "begin\tend\tnbeams\tbeam_mask\tprim_beam\tmax_snr\tbeam\n")
    }

    bool is_coincident(const Candidate* c) {
        const int64_t  sep_time   = 3;
        const uint64_t sep_filter = 4;
        const uint64_t sep_dm     = 9999;
        const float    sep_snr    = 0.30;
        const int64_t  tol        = sep_time * powf(2, max(c->filter, filter));

        // change temporal coincidence on bens suggestion 6/8/2012
        return ((abs(c->sample_idx - sample_idx) <= tol) &&
                (abs(int(c->dm_trial) - int(dm_trial)) <= sep_dm) &&
                (abs(int(c->filter) - int(filter)) <= sep_filter) &&
                ((fabsf(c->snr - snr) / (c->snr + snr)) <= sep_snr));
    }

    friend std::ostream& operator<<(std::ostream& os, const Candidate* c) {
        os << c->snr << "\t" << c->sample_idx << "\t" << c->sample_time << "\t"
           << c->filter << "\t" << c->dm_trial << "\t" << c->dm << "\t"
           << c->members << "\t" << c->begin << "\t" << c->end << "\t"
           << c->nbeams << "\t" << c->beam_mask << "\t" << c->primary_beam
           << "\t" << c->max_snr << "\t" << c->beam;
        return os;
    }
};

class CandidateChunk {
public:
    CandidateChunk() {
        first_sample = 0;
        resize(0);
        verbose = 0;
    }

    CandidateChunk(int argc, int optind, char** argv) {
        // resize internal storage
        resize(argc - optind);

        char   line[1024];
        string beam;
        int    beam_number = 0;

        for (unsigned int i = 0; i < n_beams; i++) {
            if (verbose){
                fmt::print(stderr, 
                    "CandidateChunk::CandidateChunk opening file {}\n",
                    argv[optind + i]);
            }

            // determine beam number from filename
            stringstream ss(basename(argv[optind + i]));
            getline(ss, first_sample_utc, '_');  // candidates
            getline(ss, beam, '.');              // beam number
            beam_number     = atoi(beam.c_str());
            beam_numbers[i] = beam_number;

            if (verbose){
                fmt::print(stderr, 
                    "CandidateChunk::CandidateChunk parsed beam number as {}\n",
                    beam);
            }

            std::ifstream ifs(argv[optind + i], std::ios::in);
            while (ifs.good()) {
                ifs.getline(line, 1024, '\n');
                if (!ifs.eof()) {
                    cands[i].push_back(new Candidate(line, beam_number));
                }
            }
        }
    }

    ~CandidateChunk() {
        if (verbose){
            fmt::print(stderr, "CandidateChunk::~CandidateChunk\n")
        }
        for (unsigned i = 0; i < n_beams; i++) {
            for (unsigned j = 0; j < cands[i].size(); j++)
                delete cands[i][j];
            cands[i].erase(cands[i].begin(), cands[i].end());
        }
        cands.erase(cands.begin(), cands.end());
    }

    // add beam
    void addBeam(std::string         _utc_start,
                 std::string         _first_sample_utc,
                 uint64_t            _first_sample,
                 unsigned int        beam,
                 uint64_t            num_events,
                 std::istringstream& ss) {
        unsigned int ibeam = n_beams;
        if (n_beams == 0) {
            first_sample     = _first_sample;
            first_sample_utc = _first_sample_utc;
            utc_start        = _utc_start;
        } else {
            if (first_sample != _first_sample){
                fmt::print(stderr, 
                    "CandidateChunk::addBeam sample mismatch [{} != {}]\n",
                    first_sample, _first_sample);
            }
            if (utc_start != _utc_start){
                fmt::print(stderr, "CandidateChunk::addBeam utc_start mismatch\n");
            }
        }

        // resize storage for this new beam
        resize(n_beams + 1);
        beam_numbers[ibeam] = beam;

        if (verbose > 1){
            fmt::print(stderr, 
                "CandidateChunk::addBeam resized to {} beams with beam {}\n",
                n_beams, beam);
        }

        char cand_line[1024];
        cands[ibeam].resize(num_events);
        for (unsigned ievent = 0; ievent < num_events; ievent++) {
            ss.getline(cand_line, 1024, '\n');
            cands[ibeam][ievent] = new Candidate(cand_line, beam);
        }
    }

    void resize(unsigned _n_beams) {
        n_beams = _n_beams;
        cands.resize(_n_beams);
        beam_numbers.resize(_n_beams);
    }

    unsigned int get_n_beams() const { return n_beams; }

    void compute_coincidence() {
        float    max_snr_j;
        float    snr_l;
        unsigned i, j, k, l;

        unsigned int members_tol = 3;
        unsigned int rfi_mask    = 1 << 16;
        unsigned int beam_thresh = 2;

        // compute coincidence information
        for (i = 0; i < n_beams; i++) {
            for (j = 0; j < cands[i].size(); j++) {
                max_snr_j = cands[i][j]->snr;

                if (cands[i][j]->members < members_tol)
                    continue;

                for (k = 0; k < n_beams; k++) {
                    if (i != k) {
                        for (l = 0; l < cands[k].size(); l++) {
                            snr_l = cands[k][l]->snr;

                            if (cands[k][l]->members < members_tol)
                                continue;

                            if (cands[i][j]->is_coincident(cands[k][l])) {
                                cands[i][j]->nbeams++;
                                cands[i][j]->beam_mask |= 1 << k;

                                if (cands[i][j]->nbeams >= beam_thresh + 1)
                                    cands[i][j]->beam_mask |= rfi_mask;

                                if (snr_l > max_snr_j) {
                                    cands[i][j]->primary_beam = beam_numbers[k];
                                    cands[i][j]->max_snr      = snr_l;
                                    max_snr_j                 = snr_l;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    void write_coincident_candidates() {
        std::string* filename;
        if (utc_start == "")
            filename = new std::string(first_sample_utc + "_all.cand");
        else {
            struct stat dir_stat;
            if ((stat(utc_start.c_str(), &dir_stat) == 0) &&
                (((dir_stat.st_mode) & S_IFMT) == S_IFDIR))
                filename = new std::string(utc_start + "/" + first_sample_utc +
                                           "_all.cand");
            else {
                fmt::print(stderr, 
                    "directory [{}] did not exist, not writing candidate file\n",
                    utc_start);
                return;
            }
        }
        std::ofstream ofs(filename->c_str(), std::ios::out);

        if (verbose){
            fmt::print(stderr, 
                "CandidateChunk::write_coincident_candidates: output_file={}\n",
                filename);
        }

        for (unsigned i = 0; i < n_beams; i++)
            for (unsigned j = 0; j < cands[i].size(); j++)
                ofs << cands[i][j] << endl;
        ofs.close();
    }

    // returns true if utc matches this chunk's utc_start
    bool matches(std::string utc) {
        return (first_sample_utc.compare(utc) == 0);
    }

    // return the relative age (in seconds) for the specified utc
    time_t get_relative_age(std::string utc) {
        time_t self_age = str2utctime(first_sample_utc.c_str());
        time_t utc_age  = str2utctime(utc.c_str());
        return (utc_age - self_age);
    }

private:
    std::vector<std::vector<Candidate*>> cands;
    std::vector<unsigned int>            beam_numbers;
    unsigned int                         n_beams;
    uint64_t                             first_sample;
    std::string                          first_sample_utc;
    std::string                          utc_start;
    int                                  verbose;

    time_t str2utctime(const char* str) {
        struct tm time;
        return str2utctm(&time, str);
    }

    time_t str2utctm(struct tm* time, const char* str) {
        /* append the GMT+0 timeszone information */
        char* str_utc = (char*)malloc(sizeof(char) * (strlen(str) + 4 + 1));
        sprintf(str_utc, "%s UTC", str);

        const char* format = "%Y-%m-%d-%H:%M:%S %Z";

        strptime(str_utc, format, time);

        free(str_utc);
        return timegm(time);
    }
};
