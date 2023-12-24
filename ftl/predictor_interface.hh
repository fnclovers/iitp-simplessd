#pragma once

#ifndef __SIM_CO_READ_ML_MODEL_HH__
#define __SIM_CO_READ_ML_MODEL_HH__

#include <fstream>
#include <map>

#include "cpu/cpu.hh"
#include "sim/object.hh"

namespace SimpleSSD::ML {

// Prediction result (us)
struct CoReadPrediction {
  LPN targetLPN;
  uint64_t interval;

  bool hit;

  CoReadPrediction()
      : targetLPN(UINT64_MAX), interval(0), hit(false) {}
  CoReadPrediction(LPN l)
      : targetLPN(l), interval(0), hit(false) {}
};

// Entry for idle time prediction
struct Entry {
  uint64_t issued;
  LPN slpn;
  uint32_t nlp;
};

struct WindowEntry {
  std::vector<Entry> entries;
  LPN slpn;
  uint32_t nlp;

  WindowEntry(LPN l, uint32_t n) : slpn(l), nlp(n) {}
};

class CoReadPredictor;

class AbstractMLModel : public Object {
 protected:
  CoReadPredictor *pPredictor;

  CPU::Function inferenceFstat;
  CPU::Function allocationFstat;
  CPU::Function trainingFstat;

  Event eventInferenceDone;
  Event eventAllocationDone;
  Event eventTrainingDone;

  uint64_t trainingCnt;
  uint64_t predictionRound;
  uint64_t predictionDebug;
  uint64_t BGDepth;
  uint64_t UserDepth;

  struct InferenceInput {
    LPN lpn;
    bool hit;
    std::vector<LPN> history;
    uint64_t start;
    uint64_t delayedByGC;
    uint64_t delayedByUser;

    InferenceInput(LPN l, bool h, uint64_t now, uint64_t BGDepth,
                   uint64_t UserDepth)
        : lpn(l),
          hit(h),
          start(now),
          delayedByGC(BGDepth),
          delayedByUser(UserDepth) {}

    InferenceInput(LPN l, bool h, std::vector<LPN> &&ht, uint64_t now,
                   uint64_t BGDepth, uint64_t UserDepth)
        : lpn(l),
          hit(h),
          history(std::move(ht)),
          start(now),
          delayedByGC(BGDepth),
          delayedByUser(UserDepth) {}
  };

 protected:
  std::unordered_map<uint64_t, InferenceInput> inferenceQueue;
  std::unordered_map<uint64_t, WindowEntry> trainingQueue;

 public:
  AbstractMLModel(ObjectData &o);
  virtual ~AbstractMLModel() {}

  virtual void init(CoReadPredictor *, uint64_t totalLogicalPages);

  virtual void inference(LPN, bool, uint64_t tag, bool);
  virtual void inferenceDone(uint64_t, uint64_t) = 0;
  void cancelInference(LPN);
  virtual void train(const WindowEntry &&);
  virtual void trainingDone(uint64_t, uint64_t) = 0;
  void cancelTraining(LPN);

  void storePrediction(uint64_t, CoReadPrediction &&);
  void storeHistory(WindowEntry &);

  void getStatList(std::vector<Stat> &, std::string) noexcept override {}
  void getStatValues(std::vector<double> &) noexcept override {}
  void resetStatValues() noexcept override {}

  void createCheckpoint(std::ostream &) const noexcept override {}
  void restoreCheckpoint(std::istream &) noexcept override {}
};

}  // namespace SimpleSSD::ML
#endif