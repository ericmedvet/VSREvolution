package it.units.erallab.builder.misc;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.spikingfunction.QuantizedMSN;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedMultivariateSpikingFunction;
import it.units.erallab.hmsrobots.util.Grid;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * @author eric
 */
public class QuantizedSpikingFunctionGrid implements NamedProvider<PrototypedFunctionBuilder<List<Double>, Grid<QuantizedMultivariateSpikingFunction>>> {

  private final static String INNER_SEPARATOR = "~";

  private final NamedProvider<PrototypedFunctionBuilder<List<Double>, QuantizedMultivariateSpikingFunction>> mapperBuilderProvider =
      NamedProvider.of(Map.ofEntries(
          Map.entry("qSnn", new QuantizedMSN())
      ));

  @Override
  public PrototypedFunctionBuilder<List<Double>, Grid<QuantizedMultivariateSpikingFunction>> build(Map<String, String> params) {
    PrototypedFunctionBuilder<List<Double>, QuantizedMultivariateSpikingFunction> itemBuilder = mapperBuilderProvider.build(
        params.get("iB").replace(INNER_SEPARATOR, TOKEN_SEPARATOR).replace(":", PARAM_VALUE_SEPARATOR)
    ).get();
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, Grid<QuantizedMultivariateSpikingFunction>> buildFor(Grid<QuantizedMultivariateSpikingFunction> targetFunctions) {
        return values -> {
          Grid<QuantizedMultivariateSpikingFunction> functions = Grid.create(targetFunctions);
          int c = 0;
          for (Grid.Entry<QuantizedMultivariateSpikingFunction> entry : targetFunctions) {
            if (entry.value() == null) {
              continue;
            }
            int size = itemBuilder.exampleFor(entry.value()).size();
            functions.set(
                entry.key().x(),
                entry.key().y(),
                itemBuilder.buildFor(entry.value()).apply(values.subList(c, c + size))
            );
            c = c + size;
          }
          return functions;
        };
      }

      @Override
      public List<Double> exampleFor(Grid<QuantizedMultivariateSpikingFunction> functions) {
        return Collections.nCopies(
            functions.values().stream()
                .filter(Objects::nonNull)
                .mapToInt(f -> itemBuilder.exampleFor(f).size())
                .sum(),
            0d
        );
      }
    };
  }
}
