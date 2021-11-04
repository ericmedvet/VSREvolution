package it.units.erallab.devolocomotion;

import com.google.common.base.Stopwatch;
import it.units.erallab.builder.DirectNumbersGrid;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.devofunction.*;
import it.units.erallab.builder.evolver.EvolverBuilder;
import it.units.erallab.builder.evolver.PairsTree;
import it.units.erallab.builder.evolver.TreeAndDoubles;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DevoOutcome;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DistanceBasedDevoLocomotion;
import it.units.erallab.hmsrobots.tasks.devolocomotion.TimeBasedDevoLocomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.stopcondition.FitnessEvaluations;
import it.units.malelab.jgea.core.listener.*;
import it.units.malelab.jgea.core.listener.telegram.TelegramUpdater;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.TextPlotter;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

import static it.units.erallab.devolocomotion.NamedFunctions.*;
import static it.units.erallab.hmsrobots.util.Utils.params;
import static it.units.malelab.jgea.core.listener.NamedFunctions.best;
import static it.units.malelab.jgea.core.listener.NamedFunctions.f;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author "Eric Medvet" on 2021/09/27 for VSREvolution
 */
public class Starter extends Worker {

  public Starter(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new Starter(args);
  }

  public static class DevoValidationOutcome {
    private final Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome> event;
    private final Map<String, Object> keys;
    private final DevoOutcome devoOutcome;

    public DevoValidationOutcome(Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome> event, Map<String, Object> keys, DevoOutcome devoOutcome) {
      this.event = event;
      this.keys = keys;
      this.devoOutcome = devoOutcome;
    }
  }

  @Override
  public void run() {
    //main params
    String developmentCriterion = a("devoCriterion", "distance");
    boolean distanceBasedDevelopment = developmentCriterion.startsWith("d");
    double episodeTime = d(a("episodeTime", "60"));
    double stageMaxTime = d(a("stageMaxTime", "20"));
    double stageMinDistance = d(a("stageMinDistance", "20"));
    String stringDevelopmentSchedule = a("devoSchedule", "10,20,30,45");
    List<Double> developmentSchedule = l(stringDevelopmentSchedule).stream().map(Double::parseDouble).collect(Collectors.toList());
    double validationEpisodeTime = d(a("validationEpisodeTime", Double.toString(episodeTime)));
    double validationStageMaxTime = d(a("validationStageMaxTime", Double.toString(stageMaxTime)));
    double validationStageMinDistance = d(a("validationStageMinDistance", Double.toString(stageMinDistance)));
    List<Double> validationDevelopmentSchedule = l(a("validationDevoSchedule", stringDevelopmentSchedule)).stream().map(Double::parseDouble).collect(Collectors.toList());
    int nEvals = i(a("nEvals", "800"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    int gridW = i(a("gridW", "5"));
    int gridH = i(a("gridH", "5"));
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> devoFunctionNames = l(a("devoFunction", "devoPhases-1.0-5-1<directNumGrid"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform-t+a-0.01"));
    List<String> evolverNames = l(a("evolver", "ES-16-0.35"));
    String lastFileName = a("lastFile", null);
    String bestFileName = a("bestFile", null);
    String validationFileName = a("validationFile", null);
    boolean deferred = a("deferred", "true").startsWith("t");
    List<String> serializationFlags = l(a("serialization", "")); //last,best,validation
    boolean output = a("output", "false").startsWith("t");
    String telegramBotId = a("telegramBotId", null);
    long telegramChatId = Long.parseLong(a("telegramChatId", "0"));
    List<String> validationTerrainNames = l(a("validationTerrain", "flat,downhill-30")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    //fitness function
    String fitnessFunctionName = a("fitness", "distance");
    Function<DevoOutcome, Double> fitnessFunction = getFitnessFunctionFromName(fitnessFunctionName);
    //consumers
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> keysFunctions = keysFunctions();
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> basicFunctions = basicFunctions();
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> populationFunctions = populationFunctions(fitnessFunction);
    List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> basicIndividualFunctions = basicIndividualFunctions(fitnessFunction);
    List<NamedFunction<DevoOutcome, ?>> outcomesFunctions = outcomesFunctions();
    List<NamedFunction<DevoOutcome, ?>> lastOutcomesFunctions = outcomesFunctions();
    List<NamedFunction<DevoOutcome, ?>> validationOutcomesFunctions = outcomesFunctions();
    if (serializationFlags.contains("best")) {
      List<NamedFunction<DevoOutcome, ?>> tempFunctions = new ArrayList<>(outcomesFunctions);
      tempFunctions.addAll(serializedOutcomesInformation());
      outcomesFunctions = tempFunctions;
    }
    if (serializationFlags.contains("validation")) {
      List<NamedFunction<DevoOutcome, ?>> tempFunctions = new ArrayList<>(validationOutcomesFunctions);
      tempFunctions.addAll(serializedOutcomesInformation());
      validationOutcomesFunctions = tempFunctions;
    }
    if (serializationFlags.contains("last")) {
      List<NamedFunction<DevoOutcome, ?>> tempFunctions = new ArrayList<>(lastOutcomesFunctions);
      tempFunctions.addAll(serializedOutcomesInformation());
      lastOutcomesFunctions = tempFunctions;
    }
    List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> visualIndividualFunctions = visualIndividualFunctions();
    Listener.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>> factory = Listener.Factory.deaf();
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, DevoOutcome> bestFitness = f("best.fitness", event -> Misc.first(event.getOrderedPopulation().firsts()).getFitness());
    //screen listener
    if ((bestFileName == null) || output) {
      factory = factory.and(new TabularPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(best(), visualIndividualFunctions),
          NamedFunction.then(bestFitness, outcomesFunctions())
      ))));
    }
    //file listeners
    if (lastFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(bestFitness, lastOutcomesFunctions)
      )), new File(lastFileName)
      ).onLast());
    }
    if (bestFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(bestFitness, outcomesFunctions)
      )), new File(bestFileName)
      ));
    }
    //validation listener
    if (validationFileName != null) {
      if (validationTerrainNames.isEmpty()) {
        validationTerrainNames.add(terrainNames.get(0));
      }
      Listener.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>> validationFactory = Listener.Factory.forEach(
          validation(validationTerrainNames, List.of(0),
              validationStageMinDistance, validationStageMaxTime, validationDevelopmentSchedule, validationEpisodeTime, distanceBasedDevelopment),
          new CSVPrinter<>(
              Misc.concat(List.of(
                  NamedFunction.then(f("event", (DevoValidationOutcome vo) -> vo.event), keysFunctions),
                  NamedFunction.then(f("event", (DevoValidationOutcome vo) -> vo.event), basicFunctions),
                  NamedFunction.then(f("keys", (DevoValidationOutcome vo) -> vo.keys), List.of(
                      f("validation.terrain", (Map<String, Object> map) -> map.get("validation.terrain")),
                      f("validation.seed", "%2d", (Map<String, Object> map) -> map.get("validation.seed"))
                  )),
                  NamedFunction.then(f("outcome", (DevoValidationOutcome vo) -> vo.devoOutcome), validationOutcomesFunctions))
              ),
              new File(validationFileName)
          )
      ).onLast();
      factory = factory.and(validationFactory);
    }
    //telegram listener
    if (telegramBotId != null && telegramChatId != 0) {
      factory = factory.and(new TelegramUpdater<>(List.of(
          lastEventToString(fitnessFunction),
          fitnessPlot(fitnessFunction),
          bestVideo(stageMinDistance, stageMaxTime, developmentSchedule, episodeTime, distanceBasedDevelopment)
      ), telegramBotId, telegramChatId));
    }
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Evolvers: " + evolverNames);
    L.info("Devo functions: " + devoFunctionNames);
    L.info("Sensor configs: " + targetSensorConfigNames);
    L.info("Terrains: " + terrainNames);
    L.info("Fitness: " + fitnessFunctionName);
    //start iterations
    int nOfRuns = seeds.length * terrainNames.size() * devoFunctionNames.size() * evolverNames.size();
    int counter = 0;
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String devoFunctionName : devoFunctionNames) {
          for (String targetSensorConfigName : targetSensorConfigNames) {
            for (String evolverName : evolverNames) {
              counter = counter + 1;
              final Random random = new Random(seed);
              //prepare keys
              Map<String, Object> keys = Map.ofEntries(
                  Map.entry("experiment.name", experimentName),
                  Map.entry("fitness", fitnessFunctionName),
                  Map.entry("seed", seed),
                  Map.entry("terrain", terrainName),
                  Map.entry("devo.function", devoFunctionName),
                  Map.entry("sensor.config", targetSensorConfigName),
                  Map.entry("evolver", evolverName),
                  Map.entry("episode.time", episodeTime),
                  Map.entry("stage.max.time", stageMaxTime),
                  Map.entry("stage.min.dist", stageMinDistance),
                  Map.entry("development.schedule", stringDevelopmentSchedule),
                  Map.entry("development.criterion", developmentCriterion)
              );
              //prepare target
              UnaryOperator<Robot<?>> target = r -> new Robot<>(
                  Controller.empty(),
                  RobotUtils.buildSensorizingFunction(targetSensorConfigName).apply(RobotUtils.buildShape("box-" + gridW + "x" + gridH))
              );
              //build evolver
              Evolver<?, UnaryOperator<Robot<?>>, DevoOutcome> evolver;
              try {
                evolver = buildEvolver(evolverName, devoFunctionName, target, fitnessFunction);
              } catch (ClassCastException | IllegalArgumentException e) {
                L.warning(String.format(
                    "Cannot instantiate %s for %s: %s",
                    evolverName,
                    devoFunctionName,
                    e
                ));
                continue;
              }
              Listener<Event<?, ? extends UnaryOperator<Robot<?>>, DevoOutcome>> listener = Listener.all(List.of(
                  new EventAugmenter(keys),
                  factory.build()
              ));
              if (deferred) {
                listener = listener.deferred(executorService);
              }
              //optimize
              Stopwatch stopwatch = Stopwatch.createStarted();
              L.info(String.format("Progress %s (%d/%d); Starting %s",
                  TextPlotter.horizontalBar(counter - 1, 0, nOfRuns, 8),
                  counter, nOfRuns,
                  keys
              ));
              //build task
              try {
                Collection<UnaryOperator<Robot<?>>> solutions = evolver.solve(
                    buildLocomotionTask(terrainName, stageMinDistance, stageMaxTime, developmentSchedule, episodeTime, distanceBasedDevelopment, random),
                    new FitnessEvaluations(nEvals),
                    random,
                    executorService,
                    listener
                );
                L.info(String.format("Progress %s (%d/%d); Done: %d solutions in %4ds",
                    TextPlotter.horizontalBar(counter, 0, nOfRuns, 8),
                    counter, nOfRuns,
                    solutions.size(),
                    stopwatch.elapsed(TimeUnit.SECONDS)
                ));
              } catch (Exception e) {
                L.severe(String.format("Cannot complete %s due to %s",
                    keys,
                    e
                ));
                e.printStackTrace(); // TODO possibly to be removed
              }
            }
          }
        }
      }
    }
    factory.shutdown();
  }

  public static Function<UnaryOperator<Robot<?>>, DevoOutcome> buildLocomotionTask(
      String terrainName, double stageMinDistance, double stageMaxT, List<Double> developmentSchedule, double maxT, boolean distanceBasedDevelopment, Random random) {
    if (!terrainName.contains("-rnd")) {
      Task<UnaryOperator<Robot<?>>, DevoOutcome> devoLocomotion;
      if (distanceBasedDevelopment) {
        devoLocomotion = new DistanceBasedDevoLocomotion(
            stageMinDistance, stageMaxT, maxT,
            Locomotion.createTerrain(terrainName),
            it.units.erallab.locomotion.Starter.PHYSICS_SETTINGS
        );
      } else {
        devoLocomotion = new TimeBasedDevoLocomotion(
            developmentSchedule, maxT,
            Locomotion.createTerrain(terrainName),
            it.units.erallab.locomotion.Starter.PHYSICS_SETTINGS
        );
      }
      return Misc.cached(devoLocomotion, it.units.erallab.locomotion.Starter.CACHE_SIZE);
    }
    return r -> {
      Task<UnaryOperator<Robot<?>>, DevoOutcome> devoLocomotion;
      if (distanceBasedDevelopment) {
        devoLocomotion = new DistanceBasedDevoLocomotion(
            stageMinDistance, stageMaxT, maxT,
            Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
            it.units.erallab.locomotion.Starter.PHYSICS_SETTINGS
        );
      } else {
        devoLocomotion = new TimeBasedDevoLocomotion(
            developmentSchedule, maxT,
            Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
            it.units.erallab.locomotion.Starter.PHYSICS_SETTINGS
        );
      }
      return devoLocomotion.apply(r);
    };
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static Evolver<?, UnaryOperator<Robot<?>>, DevoOutcome> buildEvolver(String evolverName, String devoFunctionName, UnaryOperator<Robot<?>> target, Function<DevoOutcome, Double> outcomeMeasure) {
    PrototypedFunctionBuilder<?, ?> devoFunctionBuilder = null;
    for (String piece : devoFunctionName.split(it.units.erallab.locomotion.Starter.MAPPER_PIPE_CHAR)) {
      if (devoFunctionBuilder == null) {
        devoFunctionBuilder = getDevoFunctionByName(piece);
      } else {
        devoFunctionBuilder = devoFunctionBuilder.compose((PrototypedFunctionBuilder) getDevoFunctionByName(piece));
      }
    }
    return getEvolverBuilderFromName(evolverName).build(
        (PrototypedFunctionBuilder) devoFunctionBuilder,
        target,
        PartialComparator.from(Double.class).comparing(outcomeMeasure).reversed()
    );
  }

  private static PrototypedFunctionBuilder<?, ?> getDevoFunctionByName(String name) {
    String devoHomoMLP = "devoHomoMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<nSignals>\\d+)-(?<nInitial>\\d+)-(?<nStep>\\d+)" +
        "(-(?<cStep>\\d+(\\.\\d+)?))?";
    String devoRandomHomoMLP = "devoRndHomoMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<nSignals>\\d+)-(?<nInitial>\\d+)-(?<nStep>\\d+)";
    String devoRandomAdditionHomoMLP = "devoRndAddHomoMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<nSignals>\\d+)-(?<nInitial>\\d+)-(?<nStep>\\d+)";
    String devoCondHomoMLP = "devoCondHomoMLP-(?<ratio>\\d+(\\.\\d+)?)" +
        "-(?<nLayers>\\d+)-(?<nSignals>\\d+)" +
        "-(?<selFunc>(areaRatioEnergy|areaRatio))-(?<maxFirst>(t|f))" +
        "-(?<nInitial>\\d+)-(?<nStep>\\d+)";
    String devoTreeHomoMLP = "devoTreeHomoMLP-(?<ratio>\\d+(\\.\\d+)?)" +
        "-(?<nLayers>\\d+)-(?<nSignals>\\d+)" +
        "-(?<nInitial>\\d+)-(?<nStep>\\d+)" +
        "(-(?<cStep>\\d+(\\.\\d+)?))?";
    String devoCondTreeHomoMLP = "devoCondTreeHomoMLP-(?<ratio>\\d+(\\.\\d+)?)" +
        "-(?<nLayers>\\d+)-(?<nSignals>\\d+)" +
        "-(?<selFunc>(areaRatioEnergy|areaRatio))-(?<maxFirst>(t|f))" +
        "-(?<nInitial>\\d+)-(?<nStep>\\d+)" +
        "(-(?<cStep>\\d+(\\.\\d+)?))?";
    String devoTreePhases = "devoTreePhases-(?<f>\\d+(\\.\\d+)?)-(?<nInitial>\\d+)-(?<nStep>\\d+)" +
        "(-(?<cStep>\\d+(\\.\\d+)?))?";
    String fixedPhases = "devoPhases-(?<f>\\d+(\\.\\d+)?)-(?<nInitial>\\d+)-(?<nStep>\\d+)" +
        "(-(?<cStep>\\d+(\\.\\d+)?))?";
    String directNumGrid = "directNumGrid";
    String devoCAHomoMLP = "devoCAHomoMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<nSignals>\\d+)" +
        "-(?<caRatio>\\d+(\\.\\d+)?)-(?<caNLayers>\\d+)" +
        "-(?<nInitial>\\d+)-(?<nStep>\\d+)" +
        "(-(?<cStep>\\d+(\\.\\d+)?))?";
    Map<String, String> params;
    //devo functions
    if ((params = params(fixedPhases, name)) != null) {
      String controllerStep = params.get("cStep");
      return new DevoPhasesValues(
          Double.parseDouble(params.get("f")),
          1d,
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep")),
          controllerStep == null ? 0 : Double.parseDouble(controllerStep)
      );
    }
    if ((params = params(devoTreePhases, name)) != null) {
      String controllerStep = params.get("cStep");
      return new DevoTreePhases(
          Double.parseDouble(params.get("f")),
          1d,
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep")),
          controllerStep == null ? 0 : Double.parseDouble(controllerStep)
      );
    }
    if ((params = params(devoHomoMLP, name)) != null) {
      String controllerStep = params.get("cStep");
      return new DevoHomoMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          Integer.parseInt(params.get("nSignals")),
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep")),
          controllerStep == null ? 0 : Double.parseDouble(controllerStep)
      );
    }
    if ((params = params(devoCAHomoMLP, name)) != null) {
      String controllerStep = params.get("cStep");
      return new DevoCaMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          Integer.parseInt(params.get("nSignals")),
          Double.parseDouble(params.get("caRatio")),
          Integer.parseInt(params.get("caNLayers")),
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep")),
          controllerStep == null ? 0 : Double.parseDouble(controllerStep)
      );
    }
    if ((params = params(devoRandomHomoMLP, name)) != null) {
      return new DevoRandomHomoMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          Integer.parseInt(params.get("nSignals")),
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep"))
      );
    }
    if ((params = params(devoRandomAdditionHomoMLP, name)) != null) {
      return new DevoRandomAdditionHomoMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          Integer.parseInt(params.get("nSignals")),
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep"))
      );
    }
    if ((params = params(devoTreeHomoMLP, name)) != null) {
      String controllerStep = params.get("cStep");
      return new DevoTreeHomoMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          Integer.parseInt(params.get("nSignals")),
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep")),
          controllerStep == null ? 0 : Double.parseDouble(controllerStep)
      );
    }
    if ((params = params(devoCondHomoMLP, name)) != null) {
      return new DevoConditionedHomoMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          Integer.parseInt(params.get("nSignals")),
          params.get("selFunc").equals("areaRatioEnergy") ? Voxel::getAreaRatioEnergy : Voxel::getAreaRatio,
          params.get("maxFirst").startsWith("t"),
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep"))
      );
    }
    if ((params = params(devoCondTreeHomoMLP, name)) != null) {
      String controllerStep = params.get("cStep");
      return new DevoConditionedTreeHomoMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          Integer.parseInt(params.get("nSignals")),
          params.get("selFunc").equals("areaRatioEnergy") ? Voxel::getAreaRatioEnergy : Voxel::getAreaRatio,
          params.get("maxFirst").startsWith("t"),
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep")),
          controllerStep == null ? 0 : Double.parseDouble(controllerStep)
      );
    }
    //misc
    if (params(directNumGrid, name) != null) {
      return new DirectNumbersGrid();
    }
    throw new IllegalArgumentException(String.format("Unknown devo function name: %s", name));
  }

  public static EvolverBuilder<?> getEvolverBuilderFromName(String name) {
    String treeNumGA = "treeNumGA-(?<nPop>\\d+)-(?<diversity>(t|f))";
    String treePairGA = "treePairGA-(?<nPop>\\d+)-(?<diversity>(t|f))";
    Map<String, String> params;
    if ((params = params(treeNumGA, name)) != null) {
      return new TreeAndDoubles(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0.75d,
          params.get("diversity").equals("t")
      );
    }
    if ((params = params(treePairGA, name)) != null) {
      return new PairsTree(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0.75d,
          params.get("diversity").equals("t")
      );
    }
    return it.units.erallab.locomotion.Starter.getEvolverBuilderFromName(name);
  }

  private static Function<DevoOutcome, Double> getFitnessFunctionFromName(String name) {
    String distance = "distance";
    String maxSpeed = "maxSpeed";
    if (params(distance, name) != null) {
      return devoOutcome -> devoOutcome.getLocomotionOutcomes().stream().mapToDouble(Outcome::getDistance).sum();
    }
    if (params(maxSpeed, name) != null) {
      return devoOutcome -> devoOutcome.getLocomotionOutcomes().stream()
          .map(Outcome::getVelocity)
          .filter(v -> !v.isNaN())
          .max(Double::compare).orElse(0d);
    }
    throw new IllegalArgumentException(String.format("Unknown fitness function name: %s", name));
  }

}
