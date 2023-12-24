#include "ftl/predictor_interface.hh"

#include <cmath>
#include <unordered_set>
#include <vector>

#include "cpu/cpu.hh"
#include "ftl/ml-prediction/interface/predictor.hh"
#include "ftl/ml-prediction/interface/predictor_impl.hh"
#include "ftl/ml-prediction/interface/training_impl.hh"
#include "ftl/ml-prediction/interface/workload_monitor.hh"
#include "sim/config.hh"
#include "sim/object.hh"

namespace SimpleSSD::ML {

AbstractMLModel::AbstractMLModel(ObjectData &o)
    : Object(o), trainingCnt(0), BGDepth(0), UserDepth(0) {
  inferenceFstat = readConfigUint(SimpleSSD::Section::Simulation,
                                  SimpleSSD::Config::Key::ModelLatency);
  trainingFstat = readConfigUint(SimpleSSD::Section::Simulation,
                                 SimpleSSD::Config::Key::TrainingLatency);
  eventInferenceDone = createEvent(
      [this](uint64_t t, uint64_t d) {
        UNUSED(t);
        scheduleFunction(CPU::CPUGroup::FlashTranslationLayer,
                         eventAllocationDone, d, allocationFstat);
      },
      "ML::eventInferenceDone");
  eventAllocationDone =
      createEvent([this](uint64_t t, uint64_t d) { inferenceDone(t, d); },
                  "ML::eventInferenceDone");
  eventTrainingDone =
      createEvent([this](uint64_t t, uint64_t d) { trainingDone(t, d); },
                  "ML::eventTrainingDone");
}

void AbstractMLModel::init(CoReadPredictor *pp, uint64_t) {
  pPredictor = pp;
}

void AbstractMLModel::inference(LPN lpn, bool init, uint64_t tag, bool isBG) {
  const uint64_t now = getTick();
  if (now > 0) {
    debugprint(Log::DebugID::ML,
               "inference start LPN %" PRIx64 "h | Tag %" PRIu64, lpn, tag);
  }

  uint64_t BGDepth_ = BGDepth;
  uint64_t UserDepth_ = UserDepth;
  if (isBG) {
    BGDepth++;
    BGDepth_ = UINT64_MAX;
  }
  else {
    UserDepth++;
  }

  std::vector<LPN> history;

  inferenceQueue.emplace(tag, InferenceInput(lpn, true, std::move(history), now,
                                             BGDepth_, UserDepth_));

  if (LIKELY(!init)) {
    scheduleFunction(CPU::CPUGroup::ML, eventInferenceDone, tag,
                     inferenceFstat);
  }
  else {
    inferenceDone(now, tag);
  }
}

void AbstractMLModel::train(const WindowEntry &&window) {
  auto lpn = window.slpn;
  const uint64_t trainingTag = trainingCnt;

  if (getTick() > 0) {
    debugprint(Log::DebugID::ML,
               "training start LPN %" PRIx64 "h | Tag %" PRIu64, lpn,
               trainingTag);
  }

  trainingQueue.emplace(trainingTag, std::move(window));

  trainingCnt++;

  if (getTick() > 0) {
    scheduleFunction(CPU::CPUGroup::ML, eventTrainingDone, trainingTag,
                     trainingFstat);
  }
  else {
    // init (filling)
    trainingDone(getTick(), trainingTag);
  }
}

void AbstractMLModel::cancelInference(LPN) {
  panic("inference cancel not implemented");
}

void AbstractMLModel::cancelTraining(LPN) {
  deschedule(eventTrainingDone);
}

void AbstractMLModel::storePrediction(uint64_t tag, CoReadPrediction &&pred) {
  pPredictor->predictionBuf.emplace(tag, std::move(pred));
}

void AbstractMLModel::storeHistory(WindowEntry &window) {
  pPredictor->pTable->add(window);
}

}  // namespace SimpleSSD::ML