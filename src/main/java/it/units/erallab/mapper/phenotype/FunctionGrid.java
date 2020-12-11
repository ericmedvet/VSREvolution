package it.units.erallab.mapper.phenotype;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.mapper.PrototypedFunctionBuilder;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * @author eric
 */
public class FunctionGrid implements PrototypedFunctionBuilder<List<Double>, Grid<RealFunction>> {

  private final PrototypedFunctionBuilder<List<Double>, RealFunction> itemBuilder;

  public FunctionGrid(PrototypedFunctionBuilder<List<Double>, RealFunction> itemBuilder) {
    this.itemBuilder = itemBuilder;
  }

  @Override
  public Function<List<Double>, Grid<RealFunction>> buildFor(Grid<RealFunction> targetFunctions) {
    return values -> {
      Grid<RealFunction> functions = Grid.create(targetFunctions);
      int c = 0;
      for (Grid.Entry<RealFunction> entry : targetFunctions) {
        if (entry.getValue() == null) {
          continue;
        }
        int size = itemBuilder.exampleFor(entry.getValue()).size();
        functions.set(
            entry.getX(),
            entry.getY(),
            itemBuilder.buildFor(entry.getValue()).apply(values.subList(c, c + size))
        );
        c = c + size;
      }
      return functions;
    };
  }

  @Override
  public List<Double> exampleFor(Grid<RealFunction> functions) {
    return Collections.nCopies(
        functions.values().stream()
            .filter(Objects::nonNull)
            .mapToInt(f -> itemBuilder.exampleFor(f).size())
            .sum(),
        0d
    );
  }

}
